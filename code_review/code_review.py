"""
RPA Bot Code Reviewer - Updated with VectorStore and bot_collection usage
"""
import os
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from langchain_community.vectorstores import Chroma
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
    persist_dir: str = "./chroma_db"
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

        # Initialize VectorStores instead of raw ChromaDB
        print(f"📚 Loading bot vectorstore from {config.persist_dir}")
        self.bot_vectorstore = Chroma(
            collection_name=config.bot_collection_name,
            embedding_function=self.embeddings,
            persist_directory=config.persist_dir
        )
        print(f"   - Bot collection loaded")

        print(f"📚 Loading codebase vectorstore from {config.persist_dir}")
        self.codebase_vectorstore = Chroma(
            collection_name=config.codebase_collection_name,
            embedding_function=self.embeddings,
            persist_directory=config.persist_dir
        )
        print(f"   - Codebase collection loaded\n")

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

    def _search_context(self, vectorstore, query: str, k: int = None, filter_dict: Dict = None) -> List[str]:
        """Use VectorStore similarity_search instead of raw ChromaDB query"""
        k = k or self.config.top_k_results

        try:
            # Returns Document objects
            if filter_dict:
                docs = vectorstore.similarity_search(query, k=k, filter=filter_dict)
            else:
                docs = vectorstore.similarity_search(query, k=k)

            # Extract text content
            return [doc.page_content for doc in docs]
        except Exception as e:
            print(f"⚠️  Search warning: {e}")
            return []

    def review_robot_py(self) -> Dict[str, str]:
        """Step 1: Review robot.py for workflow and layout"""
        print("📋 Reviewing robot.py...")

        # Read file directly
        robot_content = self._read_file("robot.py")

        # Search CODEBASE for workflow best practices
        workflow_patterns = self._search_context(
            self.codebase_vectorstore,
            "workflow orchestration celery tasks error handling patterns best practices"
        )

        # Search BOT for step definitions to understand available steps
        available_steps = self._search_context(
            self.bot_vectorstore,
            "def step function steps.py implementation",
            k=10
        )

        prompt = PromptTemplate(
            template="""You are an expert RPA code reviewer.
Review this robot.py file which contains the workflow orchestration.

WORKFLOW BEST PRACTICES FROM CODEBASE:
{workflow_patterns}

AVAILABLE STEPS IN THIS BOT:
{available_steps}

FILE: robot.py
{content}

Provide a review covering:
1. Workflow structure and organization
2. Are all called steps actually defined?
3. Potential risks (error handling, race conditions, resource management)
4. Use of control flow (if/while/for)
5. State management practices
6. Celery task configuration
7. Are there any orphaned/unused workflow sections?

Keep the review concise and focused on issues.
""",
            input_variables=["workflow_patterns", "available_steps", "content"]
        )

        review = self.llm.invoke(prompt.format(
            workflow_patterns="\n---\n".join(workflow_patterns[:3]),
            available_steps="\n---\n".join(available_steps[:5]),
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
            print(f"  Reviewing step: {step['name']}...", end=" ")

            # 1. Search CODEBASE for best practices
            codebase_patterns = self._search_context(
                self.codebase_vectorstore,
                f"business logic step {step['name']} error handling validation best practices",
                k=3
            )

            # 2. Search BOT for where this step is used in robot.py
            step_usage = self._search_context(
                self.bot_vectorstore,
                f"{step['name']} called invoke workflow robot.py usage",
                k=3
            )

            # 3. Search BOT for similar code (internal duplication within bot)
            internal_duplication = self._search_context(
                self.bot_vectorstore,
                step['code'][:500],
                k=3
            )

            # 4. Search CODEBASE for similar code (external duplication)
            external_duplication = self._search_context(
                self.codebase_vectorstore,
                step['code'][:500],
                k=2
            )

            review = self._review_single_step(
                step,
                self.context_manager.get_context(),
                codebase_patterns,
                step_usage,
                internal_duplication,
                external_duplication
            )

            # Only include if there are issues
            if self._has_issues(review):
                step_reviews.append({
                    "step_name": step['name'],
                    "line_number": step['line'],
                    "review": review
                })
                print("⚠️  Issues found")
            else:
                print("✅")

        # Update context with summary
        summary = f"steps.py: {len(steps)} steps reviewed, {len(step_reviews)} with issues"
        self.context_manager.add_context(summary, priority="medium")

        return {
            "file": "steps.py",
            "reviews": step_reviews,
            "total_steps": len(steps)
        }

    def _review_single_step(self, step: Dict, context: str,
                           codebase_patterns: List[str],
                           step_usage: List[str],
                           internal_dup: List[str],
                           external_dup: List[str]) -> str:
        """Review a single step function"""
        prompt = PromptTemplate(
            template="""You are reviewing a step function in an RPA bot.

CONTEXT FROM PREVIOUS ANALYSIS:
{context}

BEST PRACTICES FROM CODEBASE:
{codebase_patterns}

WHERE THIS STEP IS USED IN THE BOT:
{step_usage}

SIMILAR CODE IN THIS BOT (check for internal duplication):
{internal_dup}

SIMILAR CODE IN CODEBASE (check if better version exists):
{external_dup}

STEP TO REVIEW:
Function: {name}
Line: {line}
Code:
{code}

Review for:
1. Is this step actually called in the workflow? (check usage above)
2. Is it duplicated elsewhere in THIS bot? (internal duplication)
3. Does better code exist in the codebase? (external duplication)
4. Business logic correctness
5. Error handling and validation
6. State management
7. Data validation and type checking

Only report if there are actual issues. Be concise.
""",
            input_variables=["context", "codebase_patterns", "step_usage",
                           "internal_dup", "external_dup", "name", "line", "code"]
        )

        review = self.llm.invoke(prompt.format(
            context=context[:800],
            codebase_patterns="\n---\n".join(codebase_patterns[:2]) if codebase_patterns else "None found",
            step_usage="\n---\n".join(step_usage[:2]) if step_usage else "⚠️ No usage found - possibly orphaned code",
            internal_dup="\n---\n".join([d for d in internal_dup if step['name'] not in d][:2]) if internal_dup else "None",
            external_dup="\n---\n".join(external_dup[:2]) if external_dup else "None",
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
            print(f"  Reviewing {block['type']}: {block['name']}...", end=" ")

            # 1. Search CODEBASE for technical standards
            tech_standards = self._search_context(
                self.codebase_vectorstore,
                f"{block['type']} {block['name']} security error handling logging resource management",
                k=3
            )

            # 2. Search BOT for where this utility is used
            util_usage = self._search_context(
                self.bot_vectorstore,
                f"{block['name']} import from utils usage called",
                k=3
            )

            # 3. Search BOT for internal duplication
            internal_duplication = self._search_context(
                self.bot_vectorstore,
                block['code'][:500],
                k=3
            )

            # 4. Search CODEBASE for external duplication
            external_duplication = self._search_context(
                self.codebase_vectorstore,
                block['code'][:500],
                k=2
            )

            review = self._review_util_block(
                block,
                self.context_manager.get_context(),
                tech_standards,
                util_usage,
                internal_duplication,
                external_duplication
            )

            if self._has_issues(review):
                reviews.append({
                    "type": block['type'],
                    "name": block['name'],
                    "line_number": block['line'],
                    "review": review
                })
                print("⚠️  Issues found")
            else:
                print("✅")

        return {
            "file": filename,
            "reviews": reviews,
            "total_blocks": len(code_blocks)
        }

    def _review_util_block(self, block: Dict, context: str,
                          tech_standards: List[str],
                          util_usage: List[str],
                          internal_dup: List[str],
                          external_dup: List[str]) -> str:
        """Review a utility function or class"""
        prompt = PromptTemplate(
            template="""You are reviewing technical utility code in an RPA bot.

CONTEXT:
{context}

TECHNICAL STANDARDS FROM CODEBASE:
{tech_standards}

WHERE THIS UTILITY IS USED IN THE BOT:
{util_usage}

SIMILAR CODE IN THIS BOT (check for internal duplication):
{internal_dup}

SIMILAR CODE IN CODEBASE (check if better version exists):
{external_dup}

CODE TO REVIEW:
Type: {type}
Name: {name}
Line: {line}
Code:
{code}

Review for:
1. Is this utility actually used? (check usage above)
2. Is it duplicated elsewhere in THIS bot?
3. Does better code exist in the codebase?
4. Error handling and logging
5. Resource management (DB connections, files, browser sessions)
6. Security (credentials, SQL injection, XSS)
7. Performance and efficiency
8. Type hints and documentation
9. PostgreSQL query safety

Focus on technical standards. Be concise, report only issues.
""",
            input_variables=["context", "tech_standards", "util_usage",
                           "internal_dup", "external_dup", "type", "name", "line", "code"]
        )

        review = self.llm.invoke(prompt.format(
            context=context[:800],
            tech_standards="\n---\n".join(tech_standards[:2]) if tech_standards else "None found",
            util_usage="\n---\n".join(util_usage[:2]) if util_usage else "⚠️ No usage found - possibly orphaned code",
            internal_dup="\n---\n".join([d for d in internal_dup if block['name'] not in d][:2]) if internal_dup else "None",
            external_dup="\n---\n".join(external_dup[:2]) if external_dup else "None",
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
            'should', 'must', 'need to', 'warning', 'concern', 'duplicate',
            'orphaned', 'unused', 'not found', 'no usage'
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
2. Critical issues by priority:
   - Security vulnerabilities
   - Orphaned/unused code
   - Code duplication (internal and external)
   - Missing error handling
   - Other issues
3. Recommendations prioritized by severity
4. Statistics (total issues, by category)

Be concise and actionable. Include file names, function/class names, and line numbers.
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
    # CONFIGURATION
    # ========================================
    config = ReviewConfig(
        bot_path="./rpa_bot",           # Path to your bot code
        codebase_path="./rpa_codebase", # Path to reference codebase
        persist_dir="./chroma_db",      # ChromaDB storage location
        bot_collection_name="bot",
        codebase_collection_name="codebase",
        ollama_host="http://localhost:11434",
        chat_model="py-chat",
        embed_model="py-embed"
    )

    print("🤖 Starting RPA Bot Code Review")
    print(f"📁 Bot path: {config.bot_path}")
    print(f"📁 Codebase path: {config.codebase_path}")
    print(f"💾 ChromaDB persist: {config.persist_dir}\n")

    try:
        reviewer = RPABotReviewer(config)
    except Exception as e:
        print(f"❌ Error initializing reviewer: {e}")
        print("\nMake sure:")
        print("1. ChromaDB collections exist in ./chroma_db")
        print("2. Ollama is running with py-chat and py-embed models")
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
    print("\n" + "="*80)
    print("FINAL REPORT")
    print("="*80)
    print(final_report)

    # Save to file
    with open("rpa_bot_review.md", "w") as f:
        f.write(f"# RPA Bot Code Review\n\n")
        f.write(f"**Bot Path:** {config.bot_path}\n")
        f.write(f"**Codebase Path:** {config.codebase_path}\n\n")

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