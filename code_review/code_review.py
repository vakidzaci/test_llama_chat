"""
RPA Bot Code Reviewer - Reads files directly, uses ChromaDB for context search
"""
import os
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
import chromadb
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import PromptTemplate


@dataclass
class ReviewConfig:
    """Configuration for the reviewer"""
    # File paths
    bot_path: str  # Path to bot code folder
    codebase_path: str  # Path to reference codebase folder

    # ChromaDB settings
    chromadb_host: str = "localhost"
    chromadb_port: int = 8000
    bot_collection_name: str = "bot"
    codebase_collection_name: str = "codebase"

    # Ollama settings
    ollama_host: str = "http://localhost:11434"
    chat_model: str = "py-chat"
    embed_model: str = "py-embed"

    # Review settings
    max_context_tokens: int = 4000
    top_k_results: int = 5


class ContextManager:
    """Manages context size to prevent explosion"""

    def __init__(self, max_tokens: int = 4000):
        self.max_tokens = max_tokens
        self.context_history = []

    def add_context(self, new_context: str, priority: str = "medium") -> str:
        """Add new context and trim if needed"""
        self.context_history.append({
            "content": new_context,
            "priority": priority,
            "tokens": len(new_context) // 4  # Rough estimation
        })
        return self._get_trimmed_context()

    def _get_trimmed_context(self) -> str:
        """Get context trimmed to max_tokens"""
        priority_order = {"high": 3, "medium": 2, "low": 1}
        sorted_contexts = sorted(
            self.context_history,
            key=lambda x: priority_order.get(x["priority"], 0),
            reverse=True
        )

        result = []
        total_tokens = 0

        for ctx in sorted_contexts:
            if total_tokens + ctx["tokens"] <= self.max_tokens:
                result.append(ctx["content"])
                total_tokens += ctx["tokens"]

        return "\n\n".join(result)

    def get_context(self) -> str:
        return self._get_trimmed_context()


class RPABotReviewer:
    """Main reviewer class for RPA bot code"""

    def __init__(self, config: ReviewConfig):
        self.config = config
        self.bot_path = Path(config.bot_path)
        self.codebase_path = Path(config.codebase_path)

        # Initialize Ollama LLM
        self.llm = Ollama(
            model=config.chat_model,
            base_url=config.ollama_host
        )

        self.embeddings = OllamaEmbeddings(
            model=config.embed_model,
            base_url=config.ollama_host
        )

        # Initialize ChromaDB client
        print(f"🔌 Connecting to ChromaDB at {config.chromadb_host}:{config.chromadb_port}")
        self.chroma_client = chromadb.HttpClient(
            host=config.chromadb_host,
            port=config.chromadb_port
        )

        # Get ChromaDB collections
        print(f"📚 Loading collection: {config.bot_collection_name}")
        self.bot_collection = self.chroma_client.get_collection(config.bot_collection_name)
        print(f"   - Bot collection has {self.bot_collection.count()} items")

        print(f"📚 Loading collection: {config.codebase_collection_name}")
        self.codebase_collection = self.chroma_client.get_collection(config.codebase_collection_name)
        print(f"   - Codebase collection has {self.codebase_collection.count()} items\n")

        self.context_manager = ContextManager(config.max_context_tokens)

    def _read_file(self, filename: str) -> str:
        """Read file directly from bot path"""
        file_path = self.bot_path / filename

        # Also try common variations
        if not file_path.exists():
            if filename == "utils.py":
                file_path = self.bot_path / "util.py"

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {filename} in {self.bot_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _search_context(self, collection, query: str, n_results: int = None) -> List[str]:
        """Use ChromaDB for vector search to get relevant context"""
        n_results = n_results or self.config.top_k_results

        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )

        return results['documents'][0] if results['documents'] else []

    def review_robot_py(self) -> Dict[str, str]:
        """Step 1: Review robot.py for workflow and layout"""
        print("📋 Reviewing robot.py...")

        # Read file directly
        robot_content = self._read_file("robot.py")

        # Get relevant context from ChromaDB about workflow patterns
        workflow_context = self._search_context(
            self.codebase_collection,
            "workflow orchestration celery tasks error handling patterns"
        )

        prompt = PromptTemplate(
            template="""You are an expert RPA code reviewer.
Review this robot.py file which contains the workflow orchestration.

BEST PRACTICES FROM CODEBASE:
{context}

FILE: robot.py
{content}

Provide a review covering:
1. Workflow structure and organization
2. Potential risks (error handling, race conditions, resource management)
3. Use of control flow (if/while/for)
4. State management practices
5. Celery task configuration

Keep the review concise and focused on issues.
""",
            input_variables=["context", "content"]
        )

        review = self.llm.invoke(prompt.format(
            context="\n".join(workflow_context[:3]),
            content=robot_content[:8000]
        ))

        # Add summary to context for next steps
        summary = f"robot.py overview: {review[:500]}..."
        self.context_manager.add_context(summary, priority="high")

        return {
            "file": "robot.py",
            "review": review,
            "full_content": robot_content
        }

    def review_steps_py(self) -> Dict[str, List[Dict]]:
        """Step 2: Review steps.py with context from robot.py"""
        print("📋 Reviewing steps.py...")

        # Read file directly
        steps_content = self._read_file("steps.py")

        # Extract individual steps
        steps = self._extract_code_blocks(steps_content, block_type="function")

        step_reviews = []

        for step in steps:
            # Use ChromaDB to find relevant patterns from codebase
            pattern_query = f"business logic {step['name']} error handling validation"
            codebase_patterns = self._search_context(
                self.codebase_collection,
                pattern_query,
                n_results=3
            )

            # Check for code duplication using vector search
            duplication_query = step['code'][:500]  # Use actual code for similarity search
            similar_code = self._search_context(
                self.codebase_collection,
                duplication_query,
                n_results=2
            )

            review = self._review_single_step(
                step,
                self.context_manager.get_context(),
                codebase_patterns,
                similar_code
            )

            # Only include if there are issues
            if self._has_issues(review):
                step_reviews.append({
                    "step_name": step['name'],
                    "line_number": step['line'],
                    "review": review
                })

        # Update context with summary
        summary = f"steps.py: {len(steps)} steps reviewed, {len(step_reviews)} with issues"
        self.context_manager.add_context(summary, priority="medium")

        return {
            "file": "steps.py",
            "reviews": step_reviews,
            "total_steps": len(steps)
        }

    def _review_single_step(self, step: Dict, context: str,
                            patterns: List[str], similar_code: List[str]) -> str:
        """Review a single step function"""
        prompt = PromptTemplate(
            template="""You are reviewing a step function in an RPA bot.

CONTEXT FROM PREVIOUS ANALYSIS:
{context}

BEST PRACTICES PATTERNS:
{patterns}

SIMILAR CODE IN CODEBASE (check for duplication):
{similar}

STEP TO REVIEW:
Function: {name}
Line: {line}
Code:
{code}

Review for:
1. Business logic correctness
2. Error handling and validation
3. State management
4. Code duplication (if similar code exists in codebase, suggest using it)
5. Data validation and type checking

Only report if there are actual issues. Be concise.
""",
            input_variables=["context", "patterns", "similar", "name", "line", "code"]
        )

        review = self.llm.invoke(prompt.format(
            context=context[:1000],
            patterns="\n---\n".join(patterns),
            similar="\n---\n".join(similar_code),
            name=step['name'],
            line=step['line'],
            code=step['code'][:2000]
        ))

        return review

    def review_utils_py(self) -> Dict[str, List[Dict]]:
        """Step 3: Review utils.py with technical standards"""
        print("📋 Reviewing utils.py...")

        # Read file directly (try both utils.py and util.py)
        try:
            utils_content = self._read_file("utils.py")
            filename = "utils.py"
        except FileNotFoundError:
            utils_content = self._read_file("util.py")
            filename = "util.py"

        # Extract classes and functions
        code_blocks = self._extract_code_blocks(utils_content, block_type="both")

        reviews = []

        for block in code_blocks:
            # Search for technical standards from codebase
            tech_query = f"{block['type']} {block['name']} security error handling logging"
            tech_standards = self._search_context(
                self.codebase_collection,
                tech_query,
                n_results=3
            )

            # Check for code duplication
            dup_query = block['code'][:500]
            similar_code = self._search_context(
                self.codebase_collection,
                dup_query,
                n_results=2
            )

            review = self._review_util_block(
                block,
                self.context_manager.get_context(),
                tech_standards,
                similar_code
            )

            if self._has_issues(review):
                reviews.append({
                    "type": block['type'],
                    "name": block['name'],
                    "line_number": block['line'],
                    "review": review
                })

        return {
            "file": filename,
            "reviews": reviews,
            "total_blocks": len(code_blocks)
        }

    def _review_util_block(self, block: Dict, context: str,
                           standards: List[str], similar_code: List[str]) -> str:
        """Review a utility function or class"""
        prompt = PromptTemplate(
            template="""You are reviewing technical utility code in an RPA bot.

CONTEXT:
{context}

TECHNICAL STANDARDS FROM CODEBASE:
{standards}

SIMILAR CODE (check for duplication):
{similar}

CODE TO REVIEW:
Type: {type}
Name: {name}
Line: {line}
Code:
{code}

Review for:
1. Error handling and logging
2. Resource management (DB connections, files, browser sessions)
3. Security (credentials, SQL injection, XSS)
4. Code reusability and duplication
5. Performance and efficiency
6. Type hints and documentation
7. PostgreSQL query safety

Focus on technical standards. Be concise, report only issues.
""",
            input_variables=["context", "standards", "similar", "type", "name", "line", "code"]
        )

        review = self.llm.invoke(prompt.format(
            context=context[:800],
            standards="\n---\n".join(standards),
            similar="\n---\n".join(similar_code),
            type=block['type'],
            name=block['name'],
            line=block['line'],
            code=block['code'][:2000]
        ))

        return review

    def _extract_code_blocks(self, content: str, block_type: str = "function") -> List[Dict]:
        """Extract functions/classes from code"""
        blocks = []
        lines = content.split('\n')

        current_block = None
        indent_level = 0

        for i, line in enumerate(lines, 1):
            stripped = line.lstrip()

            # Detect function
            if block_type in ["function", "both"] and stripped.startswith('def '):
                if current_block:
                    blocks.append(current_block)

                func_name = stripped.split('(')[0].replace('def ', '').strip()
                current_block = {
                    'type': 'function',
                    'name': func_name,
                    'line': i,
                    'code': line + '\n',
                    'indent': len(line) - len(stripped)
                }
                indent_level = len(line) - len(stripped)

            # Detect class
            elif block_type in ["class", "both"] and stripped.startswith('class '):
                if current_block:
                    blocks.append(current_block)

                class_name = stripped.split('(')[0].split(':')[0].replace('class ', '').strip()
                current_block = {
                    'type': 'class',
                    'name': class_name,
                    'line': i,
                    'code': line + '\n',
                    'indent': len(line) - len(stripped)
                }
                indent_level = len(line) - len(stripped)

            # Continue current block
            elif current_block:
                line_indent = len(line) - len(stripped)
                if stripped == '' or line_indent > indent_level:
                    current_block['code'] += line + '\n'
                else:
                    blocks.append(current_block)
                    current_block = None

        if current_block:
            blocks.append(current_block)

        return blocks

    def _has_issues(self, review: str) -> bool:
        """Check if review contains actual issues"""
        review_lower = review.lower()
        issue_keywords = [
            'risk', 'issue', 'problem', 'error', 'missing', 'incorrect',
            'vulnerability', 'duplication', 'repeated', 'security',
            'should', 'must', 'need to', 'warning', 'concern', 'duplicate'
        ]
        return any(keyword in review_lower for keyword in issue_keywords)

    def generate_final_report(self, robot_review: Dict,
                              steps_review: Dict, utils_review: Dict) -> str:
        """Generate consolidated final report"""
        print("📝 Generating final report...")

        prompt = PromptTemplate(
            template="""Consolidate these code reviews into a final summary report.

ROBOT.PY REVIEW:
{robot}

STEPS.PY REVIEWS ({steps_count} issues found out of {steps_total} steps):
{steps}

UTILS.PY REVIEWS ({utils_count} issues found out of {utils_total} blocks):
{utils}

Create a summary with:
1. Executive summary (3-4 sentences)
2. Critical issues (with file, function/class name, line numbers)
3. Recommendations prioritized by severity
4. Statistics (total issues, by category)

Be concise and actionable.
""",
            input_variables=["robot", "steps", "steps_count", "steps_total",
                             "utils", "utils_count", "utils_total"]
        )

        steps_summary = "\n".join([
            f"- {r['step_name']} (line {r['line_number']}): {r['review'][:200]}..."
            for r in steps_review['reviews'][:10]
        ]) if steps_review['reviews'] else "No issues found"

        utils_summary = "\n".join([
            f"- {r['type']} {r['name']} (line {r['line_number']}): {r['review'][:200]}..."
            for r in utils_review['reviews'][:10]
        ]) if utils_review['reviews'] else "No issues found"

        report = self.llm.invoke(prompt.format(
            robot=robot_review['review'][:1500],
            steps=steps_summary,
            steps_count=len(steps_review['reviews']),
            steps_total=steps_review['total_steps'],
            utils=utils_summary,
            utils_count=len(utils_review['reviews']),
            utils_total=utils_review['total_blocks']
        ))

        return report


def main():
    """Main execution flow"""

    # ========================================
    # CONFIGURATION - UPDATE THESE PATHS
    # ========================================
    config = ReviewConfig(
        # Bot code filesystem path
        bot_path="./rpa_bot",

        # Reference codebase filesystem path
        codebase_path="./rpa_codebase",

        # ChromaDB connection (where your collections are)
        chromadb_host="localhost",
        chromadb_port=8000,
        bot_collection_name="bot",
        codebase_collection_name="codebase",

        # Ollama connection
        ollama_host="http://localhost:11434",
        chat_model="py-chat",
        embed_model="py-embed"
    )

    print("🤖 Starting RPA Bot Code Review")
    print(f"📁 Bot filesystem path: {config.bot_path}")
    print(f"📁 Codebase filesystem path: {config.codebase_path}")
    print(f"🗄️  ChromaDB: {config.chromadb_host}:{config.chromadb_port}")
    print(f"🗄️  Bot collection: '{config.bot_collection_name}'")
    print(f"🗄️  Codebase collection: '{config.codebase_collection_name}'\n")

    try:
        reviewer = RPABotReviewer(config)
    except Exception as e:
        print(f"❌ Error connecting to ChromaDB: {e}")
        print("\nMake sure:")
        print("1. ChromaDB server is running")
        print("2. Collections 'bot' and 'codebase' exist")
        print("3. Host and port are correct")
        return

    # Step 1: Review robot.py
    robot_review = reviewer.review_robot_py()

    # Step 2: Review steps.py
    steps_review = reviewer.review_steps_py()

    # Step 3: Review utils.py
    utils_review = reviewer.review_utils_py()

    # Step 4: Generate final report
    final_report = reviewer.generate_final_report(
        robot_review, steps_review, utils_review
    )

    # Display and save
    print("\n" + "=" * 80)
    print("FINAL REPORT")
    print("=" * 80)
    print(final_report)

    # Save to file
    with open("rpa_bot_review.md", "w") as f:
        f.write(f"# RPA Bot Code Review\n\n")
        f.write(f"**Bot Path:** {config.bot_path}\n\n")

        f.write(f"## 1. robot.py Review\n\n{robot_review['review']}\n\n")

        f.write(f"## 2. steps.py Review\n\n")
        f.write(f"**Total Steps:** {steps_review['total_steps']}\n")
        f.write(f"**Issues Found:** {len(steps_review['reviews'])}\n\n")

        if steps_review['reviews']:
            for r in steps_review['reviews']:
                f.write(f"### {r['step_name']} (line {r['line_number']})\n")
                f.write(f"{r['review']}\n\n")
        else:
            f.write("No issues found.\n\n")

        f.write(f"## 3. {utils_review['file']} Review\n\n")
        f.write(f"**Total Blocks:** {utils_review['total_blocks']}\n")
        f.write(f"**Issues Found:** {len(utils_review['reviews'])}\n\n")

        if utils_review['reviews']:
            for r in utils_review['reviews']:
                f.write(f"### {r['type']}: {r['name']} (line {r['line_number']})\n")
                f.write(f"{r['review']}\n\n")
        else:
            f.write("No issues found.\n\n")

        f.write(f"## Executive Summary\n\n{final_report}\n")

    print("\n✅ Review saved to rpa_bot_review.md")


if __name__ == "__main__":
    main()