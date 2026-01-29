"""
Review uncommitted changes on current branch
Compares working directory vs HEAD (last commit)
Old code from ChromaDB, new code from working directory
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
    """Configuration for diff reviewer"""
    bot_path: str = "./rpa_bot"
    persist_dir: str = "./chroma_db"
    bot_collection_name: str = "bot"
    codebase_collection_name: str = "codebase"
    ollama_host: str = "http://localhost:11434"
    chat_model: str = "py-chat"
    embed_model: str = "py-embed"


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
    """Parse git diff for current working directory changes"""

    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)

    def get_changed_files(self) -> List[str]:
        """Get list of changed Python files in working directory"""
        # Git diff for uncommitted changes
        cmd = ["git", "diff", "--name-only", "HEAD"]

        result = subprocess.run(
            cmd,
            cwd=self.repo_path,
            capture_output=True,
            text=True
        )

        files = result.stdout.strip().split('\n')
        # Filter only Python files
        return [f for f in files if f.endswith('.py') and f]

    def get_changed_line_ranges(self, file_path: str) -> List[Tuple[int, int]]:
        """Get changed line ranges for a file"""
        # Compare working directory vs HEAD
        cmd = [
            "git", "diff",
            "--unified=0",  # No context lines
            "HEAD",
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
        for line in result.stdout.split('\n'):
            if line.startswith('@@'):
                match = re.search(r'\+(\d+)(?:,(\d+))?', line)
                if match:
                    start = int(match.group(1))
                    count = int(match.group(2)) if match.group(2) else 1
                    end = start + count - 1
                    ranges.append((start, end))

        return ranges

    def get_old_file_content(self, file_path: str) -> Optional[str]:
        """Get file content from HEAD (last commit)"""
        cmd = ["git", "show", f"HEAD:{file_path}"]

        result = subprocess.run(
            cmd,
            cwd=self.repo_path,
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            return result.stdout
        return None


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
                func_lines = code.split('\n')[node.lineno - 1:node.end_lineno]
                func_code = '\n'.join(func_lines)

                functions.append({
                    'name': node.name,
                    'line_start': node.lineno,
                    'line_end': node.end_lineno,
                    'code': func_code
                })

            elif isinstance(node, ast.ClassDef):
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


class DiffReviewer:
    """Review uncommitted changes on current branch"""

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

        # Initialize VectorStores (contains committed code)
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
        self.git_parser = GitDiffParser(config.bot_path)
        self.code_parser = CodeParser()

    def get_old_function_from_git(self, file_path: str, function_name: str) -> Optional[str]:
        """Get old version of function from git HEAD"""
        old_file_content = self.git_parser.get_old_file_content(file_path)

        if not old_file_content:
            return None

        old_functions = self.code_parser.parse_functions(old_file_content)

        for func in old_functions:
            if func['name'] == function_name:
                return func['code']

        return None

    def get_new_function_code(self, file_path: str) -> str:
        """Get new version of file from working directory"""
        full_path = self.bot_path / file_path
        with open(full_path, 'r') as f:
            return f.read()

    def identify_changed_functions(self) -> List[ChangedFunction]:
        """Identify all functions that changed in working directory"""
        changed_functions = []

        # Get changed files
        changed_files = self.git_parser.get_changed_files()

        if not changed_files:
            print("\n✅ No uncommitted changes found!")
            return []

        print(f"\n📝 Found {len(changed_files)} changed files:")
        for f in changed_files:
            print(f"  - {f}")

        for file_path in changed_files:
            print(f"\n🔍 Analyzing {file_path}...")

            # Get changed line ranges
            changed_ranges = self.git_parser.get_changed_line_ranges(file_path)

            # Get new version from working directory
            try:
                new_code = self.get_new_function_code(file_path)
            except FileNotFoundError:
                print(f"  ⚠️  File not found: {file_path}")
                continue

            # Parse functions in new version
            new_functions = self.code_parser.parse_functions(new_code)

            # Get old version from git HEAD
            old_code = self.git_parser.get_old_file_content(file_path)
            old_functions = self.code_parser.parse_functions(old_code) if old_code else []

            # Build old functions dict
            old_funcs_dict = {f['name']: f for f in old_functions}

            # Find affected functions
            affected_functions = set()
            for start, end in changed_ranges:
                for func in new_functions:
                    if not (end < func['line_start'] or start > func['line_end']):
                        affected_functions.add(func['name'])

            print(f"  📌 {len(affected_functions)} functions affected: {', '.join(affected_functions)}")

            # Compare each affected function
            for func_name in affected_functions:
                new_func = next((f for f in new_functions if f['name'] == func_name), None)

                if not new_func:
                    continue

                # Get old version
                old_func = old_funcs_dict.get(func_name)

                # Determine change type
                if old_func is None:
                    change_type = "added"
                    old_code_str = None
                    print(f"    ✨ {func_name}: NEW function")
                elif old_func['code'].strip() == new_func['code'].strip():
                    # Same code, skip
                    continue
                else:
                    change_type = "modified"
                    old_code_str = old_func['code']
                    print(f"    ✏️  {func_name}: MODIFIED")

                changed_functions.append(ChangedFunction(
                    file_path=file_path,
                    function_name=func_name,
                    change_type=change_type,
                    old_code=old_code_str,
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

        # Build review prompt
        if changed_func.change_type == "added":
            prompt = self._build_new_function_prompt(
                changed_func, usage_context, codebase_patterns, similar_in_bot
            )
        else:
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
        template = PromptTemplate(
            template="""Review this NEW function in uncommitted changes.

FILE: {file_path}
FUNCTION: {function_name} (lines {line_start}-{line_end})

NEW CODE:
{new_code}

WHERE IT'S USED:
{usage}

SIMILAR PATTERNS IN CODEBASE:
{patterns}

SIMILAR CODE IN BOT:
{similar}

Review:
1. Is it used anywhere? (orphan check)
2. Duplicates existing code?
3. Error handling
4. Security issues
5. Follows best practices?

Concise bullet points only.
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
        template = PromptTemplate(
            template="""Review this MODIFIED function in uncommitted changes.

FILE: {file_path}
FUNCTION: {function_name} (lines {line_start}-{line_end})

OLD CODE (HEAD):
{old_code}

NEW CODE (Working Directory):
{new_code}

WHERE IT'S USED:
{usage}

SIMILAR PATTERNS IN CODEBASE:
{patterns}

Review the changes:
1. What changed?
2. Breaks existing callers?
3. New vulnerabilities?
4. Error handling better or worse?
5. Better patterns in codebase?

Concise bullet points only.
""",
            input_variables=["file_path", "function_name", "line_start", "line_end",
                           "old_code", "new_code", "usage", "patterns"]
        )

        return template.format(
            file_path=func.file_path,
            function_name=func.function_name,
            line_start=func.line_start,
            line_end=func.line_end,
            old_code=func.old_code or "Not found",
            new_code=func.new_code,
            usage="\n".join([d.page_content[:200] for d in usage[:2]]) if usage else "None",
            patterns="\n---\n".join([d.page_content[:300] for d in patterns[:2]]) if patterns else "None"
        )

    def review_current_changes(self) -> List[Dict]:
        """Review uncommitted changes in working directory"""
        print("="*80)
        print("🔍 REVIEWING UNCOMMITTED CHANGES")
        print("="*80)

        # Identify changed functions
        changed_functions = self.identify_changed_functions()

        if not changed_functions:
            return []

        print(f"\n📊 Total functions to review: {len(changed_functions)}\n")

        # Review each
        reviews = []
        for func in changed_functions:
            review = self.review_changed_function(func)
            reviews.append(review)

        return reviews

    def generate_report(self, reviews: List[Dict]) -> str:
        """Generate review report"""
        if not reviews:
            return "✅ No changes to review!"

        report = []
        report.append("# 🤖 Code Review - Uncommitted Changes\n")

        # Group by file
        by_file = {}
        for review in reviews:
            file = review['file']
            if file not in by_file:
                by_file[file] = []
            by_file[file].append(review)

        for file_path, file_reviews in by_file.items():
            report.append(f"## 📄 {file_path}\n")

            for rev in file_reviews:
                change_icon = "✨" if rev['change_type'] == "added" else "✏️"
                report.append(f"### {change_icon} `{rev['function']}()` (lines {rev['line_start']}-{rev['line_end']})\n")
                report.append(f"**Change Type:** {rev['change_type'].title()}\n")
                report.append(f"{rev['review']}\n")
                report.append("---\n")

        return "\n".join(report)


def main():
    """Review uncommitted changes on current branch"""
    config = ReviewConfig(
        bot_path="./rpa_bot",
        persist_dir="./chroma_db",
        chat_model="py-chat",
        embed_model="py-embed"
    )

    reviewer = DiffReviewer(config)

    # Review current uncommitted changes
    reviews = reviewer.review_current_changes()

    # Generate report
    report = reviewer.generate_report(reviews)

    # Display
    print("\n" + "="*80)
    print("📋 REVIEW REPORT")
    print("="*80)
    print(report)

    # Save
    with open("diff_review.md", "w") as f:
        f.write(report)

    print("\n✅ Review saved to diff_review.md")


if __name__ == "__main__":
    main()