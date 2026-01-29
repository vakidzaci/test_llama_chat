"""
MR-Focused Code Reviewer - Reviews only git diff changes
Old code from ChromaDB, new code from filesystem
"""
import subprocess
import re
import ast
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import PromptTemplate


@dataclass
class ReviewConfig:
    """Configuration for MR reviewer"""
    bot_path: str = "./rpa_bot"
    persist_dir: str = "./chroma_db"
    bot_collection_name: str = "bot"
    codebase_collection_name: str = "codebase"
    ollama_host: str = "http://localhost:11434"
    chat_model: str = "py-chat"
    embed_model: str = "py-embed"
    base_branch: str = "main"
    compare_branch: str = "HEAD"


@dataclass
class ChangedFunction:
    """Represents a function that changed"""
    file_path: str
    function_name: str
    change_type: str  # "added", "modified", "deleted"
    old_code: Optional[str]
    new_code: Optional[str]
    line_start: int
    line_end: int


class GitDiffParser:
    """Parse git diff to extract changed files and line ranges"""

    def __init__(self, base_branch: str, compare_branch: str, repo_path: str):
        self.base_branch = base_branch
        self.compare_branch = compare_branch
        self.repo_path = Path(repo_path)

    def get_changed_files(self) -> List[str]:
        """Get list of changed Python files"""
        cmd = [
            "git", "diff",
            "--name-only",
            f"{self.base_branch}...{self.compare_branch}"
        ]

        result = subprocess.run(
            cmd,
            cwd=self.repo_path,
            capture_output=True,
            text=True
        )

        files = result.stdout.strip().split('\n')
        # Filter only Python files
        return [f for f in files if f.endswith('.py')]

    def get_changed_line_ranges(self, file_path: str) -> List[Tuple[int, int]]:
        """Get changed line ranges for a file"""
        cmd = [
            "git", "diff",
            "--unified=0",  # No context lines
            f"{self.base_branch}...{self.compare_branch}",
            "--", file_path
        ]

        result = subprocess.run(
            cmd,
            cwd=self.repo_path,
            capture_output=True,
            text=True
        )

        ranges = []
        # Parse diff output for line ranges
        # Format: @@ -old_start,old_count +new_start,new_count @@
        for line in result.stdout.split('\n'):
            if line.startswith('@@'):
                match = re.search(r'\+(\d+)(?:,(\d+))?', line)
                if match:
                    start = int(match.group(1))
                    count = int(match.group(2)) if match.group(2) else 1
                    end = start + count - 1
                    ranges.append((start, end))

        return ranges


class CodeParser:
    """Parse Python code to extract functions and classes"""

    @staticmethod
    def parse_functions(code: str) -> List[Dict]:
        """Extract all functions with their line numbers"""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []

        functions = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Get function code
                func_lines = code.split('\n')[node.lineno - 1:node.end_lineno]
                func_code = '\n'.join(func_lines)

                functions.append({
                    'name': node.name,
                    'line_start': node.lineno,
                    'line_end': node.end_lineno,
                    'code': func_code
                })

            elif isinstance(node, ast.ClassDef):
                # Get methods within class
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        method_lines = code.split('\n')[item.lineno - 1:item.end_lineno]
                        method_code = '\n'.join(method_lines)

                        functions.append({
                            'name': f"{node.name}.{item.name}",
                            'line_start': item.lineno,
                            'line_end': item.end_lineno,
                            'code': method_code
                        })

        return functions

    @staticmethod
    def find_function_at_line(functions: List[Dict], line_num: int) -> Optional[Dict]:
        """Find which function contains a given line number"""
        for func in functions:
            if func['line_start'] <= line_num <= func['line_end']:
                return func
        return None


class MRCodeReviewer:
    """Review only changed code in a merge request"""

    def __init__(self, config: ReviewConfig):
        self.config = config
        self.bot_path = Path(config.bot_path)

        # Initialize LLM
        print(f"🤖 Initializing LLM: {config.chat_model}")
        self.llm = Ollama(
            model=config.chat_model,
            base_url=config.ollama_host
        )

        # Initialize embeddings
        print(f"🔧 Initializing embeddings: {config.embed_model}")
        self.embeddings = OllamaEmbeddings(
            model=config.embed_model,
            base_url=config.ollama_host
        )

        # Initialize VectorStores (contains old/main branch code)
        print(f"📚 Loading vectorstores from {config.persist_dir}")
        self.bot_vectorstore = Chroma(
            collection_name=config.bot_collection_name,
            embedding_function=self.embeddings,
            persist_directory=config.persist_dir
        )

        self.codebase_vectorstore = Chroma(
            collection_name=config.codebase_collection_name,
            embedding_function=self.embeddings,
            persist_directory=config.persist_dir
        )

        # Initialize git parser
        self.git_parser = GitDiffParser(
            config.base_branch,
            config.compare_branch,
            config.bot_path
        )

        self.code_parser = CodeParser()

    def get_old_function_code(self, file_path: str, function_name: str) -> Optional[str]:
        """Get old version of function from ChromaDB"""
        # Query vectorstore for this specific function
        results = self.bot_vectorstore.similarity_search(
            query=f"function {function_name} in {file_path}",
            k=10,
            filter={"filename": Path(file_path).name}
        )

        # Find the matching function by name
        for doc in results:
            if function_name in doc.page_content:
                # Try to extract just this function
                # (ChromaDB might have the full function or a chunk)
                if f"def {function_name}" in doc.page_content:
                    return doc.page_content

        return None

    def get_new_function_code(self, file_path: str) -> str:
        """Get new version of file from filesystem"""
        full_path = self.bot_path / file_path
        with open(full_path, 'r') as f:
            return f.read()

    def identify_changed_functions(self) -> List[ChangedFunction]:
        """Identify all functions that changed in the diff"""
        changed_functions = []

        # Get changed files
        changed_files = self.git_parser.get_changed_files()
        print(f"\n📝 Found {len(changed_files)} changed files:")
        for f in changed_files:
            print(f"  - {f}")

        for file_path in changed_files:
            print(f"\n🔍 Analyzing {file_path}...")

            # Get changed line ranges
            changed_ranges = self.git_parser.get_changed_line_ranges(file_path)

            # Get new version of file
            try:
                new_code = self.get_new_function_code(file_path)
            except FileNotFoundError:
                print(f"  ⚠️  File not found: {file_path}")
                continue

            # Parse functions in new version
            new_functions = self.code_parser.parse_functions(new_code)

            # Find which functions were affected by changes
            affected_functions = set()
            for start, end in changed_ranges:
                for func in new_functions:
                    # Check if change overlaps with function
                    if not (end < func['line_start'] or start > func['line_end']):
                        affected_functions.add(func['name'])

            print(f"  📌 {len(affected_functions)} functions affected: {', '.join(affected_functions)}")

            # For each affected function, get old and new versions
            for func_name in affected_functions:
                # Get new version
                new_func = next((f for f in new_functions if f['name'] == func_name), None)

                if not new_func:
                    continue

                # Get old version from ChromaDB
                old_code = self.get_old_function_code(file_path, func_name)

                # Determine change type
                if old_code is None:
                    change_type = "added"
                    print(f"    ✨ {func_name}: NEW function")
                elif old_code.strip() == new_func['code'].strip():
                    # Code is the same, probably just line numbers shifted
                    continue
                else:
                    change_type = "modified"
                    print(f"    ✏️  {func_name}: MODIFIED")

                changed_functions.append(ChangedFunction(
                    file_path=file_path,
                    function_name=func_name,
                    change_type=change_type,
                    old_code=old_code,
                    new_code=new_func['code'],
                    line_start=new_func['line_start'],
                    line_end=new_func['line_end']
                ))

        return changed_functions

    def review_changed_function(self, changed_func: ChangedFunction) -> Dict[str, str]:
        """Review a single changed function"""
        print(f"  🔬 Reviewing {changed_func.function_name}...", end=" ")

        # Get context from vectorstores

        # 1. Where is this function used?
        usage_context = self.bot_vectorstore.similarity_search(
            query=f"{changed_func.function_name} called usage workflow",
            k=3
        )

        # 2. Similar code in codebase (best practices)
        codebase_patterns = self.codebase_vectorstore.similarity_search(
            query=changed_func.new_code[:500],
            k=3
        )

        # 3. Similar code in bot (duplication check)
        similar_in_bot = self.bot_vectorstore.similarity_search(
            query=changed_func.new_code[:500],
            k=3,
            filter={"filename": {"$ne": Path(changed_func.file_path).name}}
        )

        # Build review prompt based on change type
        if changed_func.change_type == "added":
            prompt = self._build_new_function_prompt(
                changed_func, usage_context, codebase_patterns, similar_in_bot
            )
        else:  # modified
            prompt = self._build_modified_function_prompt(
                changed_func, usage_context, codebase_patterns, similar_in_bot
            )

        review = self.llm.invoke(prompt)
        print("✅")

        return {
            "file": changed_func.file_path,
            "function": changed_func.function_name,
            "line_start": changed_func.line_start,
            "line_end": changed_func.line_end,
            "change_type": changed_func.change_type,
            "review": review
        }

    def _build_new_function_prompt(self, func: ChangedFunction,
                                   usage, patterns, similar) -> str:
        """Build prompt for reviewing new function"""
        template = PromptTemplate(
            template="""You are reviewing a NEW function added in a merge request.

FILE: {file_path}
FUNCTION: {function_name} (lines {line_start}-{line_end})

NEW CODE:
{new_code}

WHERE IT'S USED (if anywhere):
{usage}

SIMILAR PATTERNS IN CODEBASE:
{patterns}

SIMILAR CODE IN BOT (possible duplication):
{similar}

Review this new function for:
1. Is it actually used/called anywhere? (orphan check)
2. Does similar functionality already exist? (duplication)
3. Error handling and validation
4. Security concerns
5. Follows best practices from codebase?
6. Type hints and documentation

Format: Concise bullet points. Only mention actual issues.
""",
            input_variables=["file_path", "function_name", "line_start", "line_end",
                             "new_code", "usage", "patterns", "similar"]
        )

        return template.format(
            file_path=func.file_path,
            function_name=func.function_name,
            line_start=func.line_start,
            line_end=func.line_end,
            new_code=func.new_code,
            usage="\n".join([d.page_content[:200] for d in usage[:2]]) if usage else "⚠️ No usage found",
            patterns="\n---\n".join([d.page_content[:300] for d in patterns[:2]]) if patterns else "None",
            similar="\n---\n".join([d.page_content[:200] for d in similar[:2]]) if similar else "None"
        )

    def _build_modified_function_prompt(self, func: ChangedFunction,
                                        usage, patterns, similar) -> str:
        """Build prompt for reviewing modified function"""
        template = PromptTemplate(
            template="""You are reviewing a MODIFIED function in a merge request.

FILE: {file_path}
FUNCTION: {function_name} (lines {line_start}-{line_end})

OLD CODE (from main branch):
{old_code}

NEW CODE (in this MR):
{new_code}

WHERE IT'S USED:
{usage}

SIMILAR PATTERNS IN CODEBASE:
{patterns}

SIMILAR CODE IN BOT:
{similar}

Review the changes:
1. What changed and why might it have changed?
2. Does the change break existing callers?
3. Are new vulnerabilities introduced?
4. Is error handling improved or degraded?
5. Better patterns available in codebase?
6. Any side effects on dependent code?

Format: Concise bullet points. Only mention actual issues or improvements.
""",
            input_variables=["file_path", "function_name", "line_start", "line_end",
                             "old_code", "new_code", "usage", "patterns", "similar"]
        )

        return template.format(
            file_path=func.file_path,
            function_name=func.function_name,
            line_start=func.line_start,
            line_end=func.line_end,
            old_code=func.old_code or "Not found in ChromaDB",
            new_code=func.new_code,
            usage="\n".join([d.page_content[:200] for d in usage[:2]]) if usage else "Not found",
            patterns="\n---\n".join([d.page_content[:300] for d in patterns[:2]]) if patterns else "None",
            similar="\n---\n".join([d.page_content[:200] for d in similar[:2]]) if similar else "None"
        )

    def review_merge_request(self) -> List[Dict]:
        """Main entry point: review all changes in MR"""
        print("=" * 80)
        print("🔍 MERGE REQUEST CODE REVIEW")
        print("=" * 80)
        print(f"Base: {self.config.base_branch}")
        print(f"Compare: {self.config.compare_branch}")

        # Identify changed functions
        changed_functions = self.identify_changed_functions()

        if not changed_functions:
            print("\n✅ No functions changed - nothing to review!")
            return []

        print(f"\n📊 Total functions to review: {len(changed_functions)}\n")

        # Review each changed function
        reviews = []
        for func in changed_functions:
            review = self.review_changed_function(func)
            reviews.append(review)

        return reviews

    def generate_mr_report(self, reviews: List[Dict]) -> str:
        """Generate MR-friendly report"""
        if not reviews:
            return "✅ No issues found - all changes look good!"

        report = []
        report.append("# 🤖 Code Review Summary\n")

        # Group by file
        by_file = {}
        for review in reviews:
            file = review['file']
            if file not in by_file:
                by_file[file] = []
            by_file[file].append(review)

        # Generate report per file
        for file_path, file_reviews in by_file.items():
            report.append(f"## 📄 {file_path}\n")

            for rev in file_reviews:
                change_icon = "✨" if rev['change_type'] == "added" else "✏️"
                report.append(
                    f"### {change_icon} `{rev['function']}()` (lines {rev['line_start']}-{rev['line_end']})\n")
                report.append(f"**Change Type:** {rev['change_type'].title()}\n")
                report.append(f"{rev['review']}\n")
                report.append("---\n")

        return "\n".join(report)


def main():
    """Main execution"""
    config = ReviewConfig(
        bot_path="./rpa_bot",
        persist_dir="./chroma_db",
        base_branch="main",
        compare_branch="HEAD",  # or "feature/add-validation"
        chat_model="py-chat",
        embed_model="py-embed"
    )

    reviewer = MRCodeReviewer(config)

    # Review the MR
    reviews = reviewer.review_merge_request()

    # Generate report
    report = reviewer.generate_mr_report(reviews)

    # Display
    print("\n" + "=" * 80)
    print("📋 REVIEW REPORT")
    print("=" * 80)
    print(report)

    # Save to file
    with open("mr_review.md", "w") as f:
        f.write(report)

    print("\n✅ Review saved to mr_review.md")


if __name__ == "__main__":
    main()