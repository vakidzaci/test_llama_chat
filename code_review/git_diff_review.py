"""
LLM-Driven Code Reviewer
Uses LLM to understand functionality, not regex pattern matching
"""
import subprocess
import ast
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import PromptTemplate


@dataclass
class ReviewConfig:
    bot_path: str = "./rpa_bot"
    codebase_path: str = "./rpa_codebase"
    persist_dir: str = "./chroma_db"
    bot_collection_name: str = "bot"
    codebase_collection_name: str = "codebase"
    ollama_host: str = "http://localhost:11434"
    chat_model: str = "py-chat"
    embed_model: str = "py-embed"
    comparison_mode: str = "uncommitted"
    base_branch: str = "main"


@dataclass
class ChangedFunction:
    file_path: str
    function_name: str
    change_type: str
    old_code: Optional[str]
    new_code: Optional[str]
    line_start: int
    line_end: int


@dataclass
class FunctionAnalysis:
    """LLM's understanding of what a function does"""
    purpose: str  # What does it do?
    operations: List[str]  # What operations does it perform?
    inputs: str  # What does it need?
    outputs: str  # What does it produce?
    risks: List[str]  # What could go wrong?
    summary: str  # One-line summary


@dataclass
class DuplicationFinding:
    function_name: str
    file_path: str
    source: str  # "bot" or "codebase"
    similarity_explanation: str
    recommendation: str
    code_snippet: str


class LLMFunctionAnalyzer:
    """Use LLM to deeply understand what a function does"""

    def __init__(self, llm, embeddings):
        self.llm = llm
        self.embeddings = embeddings

    def analyze_function(self, code: str, function_name: str) -> FunctionAnalysis:
        """Ask LLM to analyze what this function does"""

        prompt = f"""Analyze this Python function and extract its PURPOSE and CHARACTERISTICS.

FUNCTION: {function_name}

CODE:
{code}

Provide a structured analysis:

1. PURPOSE (1 sentence): What problem does this solve? What is its goal?

2. OPERATIONS (bullet points): What specific operations does it perform?
   - Reading files? (Excel, CSV, PDF, etc.)
   - Database operations? (SELECT, INSERT, UPDATE)
   - API calls?
   - Data transformations?
   - Validations?
   - Web automation?

3. INPUTS: What data/parameters does it expect?

4. OUTPUTS: What does it produce/return?

5. RISKS: What could go wrong? Security issues? Error handling problems?

Format your response EXACTLY like this:
PURPOSE: <one sentence>
OPERATIONS:
- <operation 1>
- <operation 2>
INPUTS: <description>
OUTPUTS: <description>
RISKS:
- <risk 1>
- <risk 2>
"""

        response = self.llm.invoke(prompt)

        # Parse LLM response
        return self._parse_analysis(response, code, function_name)

    def _parse_analysis(self, llm_response: str, code: str, func_name: str) -> FunctionAnalysis:
        """Parse LLM's structured response"""

        lines = llm_response.split('\n')

        purpose = ""
        operations = []
        inputs_desc = ""
        outputs_desc = ""
        risks = []

        current_section = None

        for line in lines:
            line = line.strip()

            if line.startswith('PURPOSE:'):
                purpose = line.replace('PURPOSE:', '').strip()
                current_section = None
            elif line.startswith('OPERATIONS:'):
                current_section = 'operations'
            elif line.startswith('INPUTS:'):
                inputs_desc = line.replace('INPUTS:', '').strip()
                current_section = None
            elif line.startswith('OUTPUTS:'):
                outputs_desc = line.replace('OUTPUTS:', '').strip()
                current_section = None
            elif line.startswith('RISKS:'):
                current_section = 'risks'
            elif line.startswith('-') or line.startswith('•'):
                item = line.lstrip('-•').strip()
                if current_section == 'operations':
                    operations.append(item)
                elif current_section == 'risks':
                    risks.append(item)

        return FunctionAnalysis(
            purpose=purpose or f"Function {func_name}",
            operations=operations,
            inputs=inputs_desc,
            outputs=outputs_desc,
            risks=risks,
            summary=purpose[:100] if purpose else func_name
        )


class LLMDuplicationDetector:
    """Use LLM to detect functional duplicates"""

    def __init__(self, llm, embeddings, analyzer, bot_vectorstore, codebase_vectorstore):
        self.llm = llm
        self.embeddings = embeddings
        self.analyzer = analyzer
        self.bot_vectorstore = bot_vectorstore
        self.codebase_vectorstore = codebase_vectorstore

    def find_duplicates(self, new_code: str, new_name: str,
                       new_analysis: FunctionAnalysis) -> Dict[str, List[DuplicationFinding]]:
        """Find functional duplicates using LLM understanding"""

        print(f"    🔎 Searching for functional duplicates...")

        # Step 1: Use analysis to search vectorstore intelligently
        search_query = self._build_search_query(new_analysis)

        # Step 2: Get candidates from both vectorstores
        bot_candidates = self.bot_vectorstore.similarity_search(
            query=search_query,
            k=20
        )

        codebase_candidates = self.codebase_vectorstore.similarity_search(
            query=search_query,
            k=20
        )

        # Step 3: LLM compares new function to each candidate
        bot_duplicates = self._compare_with_candidates(
            new_code, new_name, new_analysis, bot_candidates, 'bot'
        )

        codebase_duplicates = self._compare_with_candidates(
            new_code, new_name, new_analysis, codebase_candidates, 'codebase'
        )

        return {
            'bot': bot_duplicates,
            'codebase': codebase_duplicates
        }

    def _build_search_query(self, analysis: FunctionAnalysis) -> str:
        """Build search query from function analysis"""
        # Combine purpose and operations into a search query
        query_parts = [analysis.purpose]
        query_parts.extend(analysis.operations[:3])  # Top 3 operations
        return " ".join(query_parts)

    def _compare_with_candidates(self, new_code: str, new_name: str,
                                 new_analysis: FunctionAnalysis,
                                 candidates: List, source: str) -> List[DuplicationFinding]:
        """Ask LLM to compare new function with each candidate"""

        duplicates = []

        for candidate in candidates[:10]:  # Limit to top 10
            cand_name = candidate.metadata.get('element_name', 'unknown')
            cand_file = candidate.metadata.get('filename', 'unknown')
            cand_code = candidate.page_content

            # Skip if same function name
            if cand_name == new_name and source == 'bot':
                continue

            # Ask LLM: Are these functionally duplicate?
            is_duplicate, explanation = self._llm_compare_functions(
                new_code, new_name, new_analysis,
                cand_code, cand_name
            )

            if is_duplicate:
                duplicates.append(DuplicationFinding(
                    function_name=cand_name,
                    file_path=cand_file,
                    source=source,
                    similarity_explanation=explanation,
                    recommendation=self._get_recommendation(source),
                    code_snippet=cand_code[:300]
                ))

        return duplicates[:5]  # Return top 5

    def _llm_compare_functions(self, code1: str, name1: str, analysis1: FunctionAnalysis,
                               code2: str, name2: str) -> Tuple[bool, str]:
        """Ask LLM if two functions are functionally duplicate"""

        prompt = f"""Compare these two functions and determine if they are FUNCTIONALLY DUPLICATE.

Functionally duplicate means: They solve the same problem or perform the same core task, even if implemented differently.

FUNCTION 1: {name1}
Purpose: {analysis1.purpose}
Operations: {', '.join(analysis1.operations)}
Code snippet:
{code1[:500]}

FUNCTION 2: {name2}
Code:
{code2[:500]}

Question: Are these two functions doing essentially the same thing?

Answer in this format:
DUPLICATE: YES or NO
EXPLANATION: <why they are or aren't duplicates>
SIMILARITY: <percentage 0-100>

Examples:
- Both read Excel files → YES, even if one uses openpyxl and one uses pandas
- Both validate email format → YES, even if different regex
- One reads Excel, one writes Excel → NO, different purposes
- One validates user input, one validates database records → MAYBE (partial overlap)
"""

        response = self.llm.invoke(prompt)

        # Parse response
        is_duplicate = 'DUPLICATE: YES' in response.upper()

        # Extract explanation
        explanation = ""
        for line in response.split('\n'):
            if line.startswith('EXPLANATION:'):
                explanation = line.replace('EXPLANATION:', '').strip()
                break

        return is_duplicate, explanation or response[:200]

    def _get_recommendation(self, source: str) -> str:
        """Get recommendation based on where duplicate was found"""
        if source == 'bot':
            return "Consider consolidating these functions to reduce code duplication"
        else:
            return "Consider using the codebase implementation instead of rewriting"


class LLMCodeReviewer:
    """Use LLM to review code for risks and issues"""

    def __init__(self, llm):
        self.llm = llm

    def review_for_risks(self, code: str, function_name: str,
                        analysis: FunctionAnalysis) -> Dict:
        """Ask LLM to review for security and reliability risks"""

        prompt = f"""Review this function for SECURITY and RELIABILITY risks.

FUNCTION: {function_name}

ANALYSIS:
- Purpose: {analysis.purpose}
- Operations: {', '.join(analysis.operations)}

CODE:
{code}

Identify:
1. SECURITY RISKS:
   - SQL injection vulnerabilities
   - Command injection
   - Hardcoded credentials or secrets
   - Unsafe file operations
   - Path traversal risks

2. RELIABILITY RISKS:
   - Poor error handling
   - Resource leaks (unclosed files, connections)
   - Race conditions
   - Infinite loop potential
   - Missing validation

3. CONFIDENTIAL DATA EXPOSURE:
   - Hardcoded passwords, API keys, tokens
   - Logged sensitive data
   - Exposed connection strings

Format response:
SECURITY:
- <issue and line reference>

RELIABILITY:
- <issue and line reference>

CONFIDENTIAL:
- <issue and line reference>

If no issues, write "NONE" for that section.
"""

        response = self.llm.invoke(prompt)

        return self._parse_review(response)

    def _parse_review(self, llm_response: str) -> Dict:
        """Parse LLM review into structured format"""

        findings = {
            'security': [],
            'reliability': [],
            'confidential': []
        }

        lines = llm_response.split('\n')
        current_section = None

        for line in lines:
            line = line.strip()

            if 'SECURITY:' in line.upper():
                current_section = 'security'
            elif 'RELIABILITY:' in line.upper():
                current_section = 'reliability'
            elif 'CONFIDENTIAL:' in line.upper():
                current_section = 'confidential'
            elif line.startswith('-') and current_section and 'NONE' not in line.upper():
                findings[current_section].append(line.lstrip('-').strip())

        return findings


class GitDiffParser:
    """Parse git diff"""

    def __init__(self, repo_path: str, mode: str = "uncommitted", base_branch: str = "main"):
        self.repo_path = Path(repo_path)
        self.mode = mode
        self.base_branch = base_branch

    def get_changed_files(self) -> List[str]:
        if self.mode == "uncommitted":
            cmd = ["git", "diff", "--name-only", "HEAD"]
        elif self.mode == "staged":
            cmd = ["git", "diff", "--name-only", "--cached"]
        elif self.mode == "last-commit":
            cmd = ["git", "diff", "--name-only", "HEAD~1", "HEAD"]
        elif self.mode == "branch":
            cmd = ["git", "diff", "--name-only", f"{self.base_branch}...HEAD"]
        else:
            cmd = ["git", "diff", "--name-only", "HEAD"]

        result = subprocess.run(
            cmd, cwd=self.repo_path, capture_output=True,
            text=True, encoding='utf-8', errors='replace'
        )

        files = result.stdout.strip().split('\n')
        return [f for f in files if f.endswith('.py') and f]

    def get_old_file_content(self, file_path: str) -> Optional[str]:
        ref = "HEAD" if self.mode in ["uncommitted", "staged"] else "HEAD~1"
        if self.mode == "branch":
            ref = self.base_branch

        result = subprocess.run(
            ["git", "show", f"{ref}:{file_path}"],
            cwd=self.repo_path, capture_output=True,
            text=True, encoding='utf-8', errors='replace'
        )
        return result.stdout if result.returncode == 0 else None


class CodeParser:
    @staticmethod
    def parse_functions(code: str) -> List[Dict]:
        try:
            tree = ast.parse(code)
        except:
            return []

        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_lines = code.split('\n')[node.lineno - 1:node.end_lineno]
                functions.append({
                    'name': node.name,
                    'line_start': node.lineno,
                    'line_end': node.end_lineno,
                    'code': '\n'.join(func_lines)
                })
        return functions


class DiffReviewer:
    """LLM-driven code reviewer"""

    def __init__(self, config: ReviewConfig):
        self.config = config
        self.bot_path = Path(config.bot_path)

        print(f"🤖 Initializing LLM: {config.chat_model}")
        self.llm = Ollama(model=config.chat_model, base_url=config.ollama_host)

        print(f"🔧 Initializing embeddings: {config.embed_model}")
        self.embeddings = OllamaEmbeddings(model=config.embed_model, base_url=config.ollama_host)

        print(f"📚 Loading vectorstores")
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

        self.git_parser = GitDiffParser(config.bot_path, config.comparison_mode, config.base_branch)
        self.code_parser = CodeParser()

        # LLM-driven analyzers
        self.function_analyzer = LLMFunctionAnalyzer(self.llm, self.embeddings)
        self.dup_detector = LLMDuplicationDetector(
            self.llm, self.embeddings, self.function_analyzer,
            self.bot_vectorstore, self.codebase_vectorstore
        )
        self.code_reviewer = LLMCodeReviewer(self.llm)

    def review_function(self, func: ChangedFunction) -> Dict:
        """Review a single function using LLM"""

        print(f"\n  🔬 Reviewing: {func.function_name}")

        # Step 1: LLM analyzes what this function does
        print(f"    🧠 LLM analyzing function purpose...")
        analysis = self.function_analyzer.analyze_function(func.new_code, func.function_name)
        print(f"       Purpose: {analysis.purpose}")

        # Step 2: LLM finds functional duplicates
        duplicates = self.dup_detector.find_duplicates(
            func.new_code, func.function_name, analysis
        )
        print(f"       Found {len(duplicates['bot'])} bot duplicates, {len(duplicates['codebase'])} codebase")

        # Step 3: LLM reviews for risks
        print(f"    🛡️  LLM reviewing for risks...")
        risks = self.code_reviewer.review_for_risks(func.new_code, func.function_name, analysis)

        # Step 4: Find usage
        usage = self.bot_vectorstore.similarity_search(
            query=f"{func.function_name} usage called",
            k=5
        )

        return {
            "file": func.file_path,
            "function": func.function_name,
            "line_start": func.line_start,
            "analysis": analysis,
            "duplicates": duplicates,
            "risks": risks,
            "usage": usage
        }

    def review_all_changes(self):
        """Review all changed functions"""

        print("="*80)
        print("🔍 LLM-DRIVEN CODE REVIEW")
        print("="*80)

        changed_files = self.git_parser.get_changed_files()

        if not changed_files:
            print("\n❌ No changed Python files found")
            print(f"Mode: {self.config.comparison_mode}")
            return []

        print(f"\n📝 Changed files: {', '.join(changed_files)}")

        all_reviews = []

        for file_path in changed_files:
            new_code = open(self.bot_path / file_path, 'r', encoding='utf-8', errors='replace').read()
            new_funcs = self.code_parser.parse_functions(new_code)

            old_code = self.git_parser.get_old_file_content(file_path)
            old_funcs = {f['name']: f for f in self.code_parser.parse_functions(old_code)} if old_code else {}

            for func in new_funcs:
                old = old_funcs.get(func['name'])
                if old and old['code'] == func['code']:
                    continue  # Unchanged

                changed_func = ChangedFunction(
                    file_path=file_path,
                    function_name=func['name'],
                    change_type="added" if not old else "modified",
                    old_code=old['code'] if old else None,
                    new_code=func['code'],
                    line_start=func['line_start'],
                    line_end=func['line_end']
                )

                review = self.review_function(changed_func)
                all_reviews.append(review)

        return all_reviews

    def generate_report(self, reviews):
        """Generate report"""
        lines = ["# 🤖 LLM-Driven Code Review\n"]

        for rev in reviews:
            lines.append(f"## {rev['file']}: `{rev['function']}()` (lines {rev['line_start']}-{rev['line_end']})\n")

            lines.append(f"**Purpose:** {rev['analysis'].purpose}\n")

            if rev['duplicates']['bot']:
                lines.append("### ⚠️ Duplicates in Bot\n")
                for dup in rev['duplicates']['bot']:
                    lines.append(f"- **{dup.function_name}** in {dup.file_path}")
                    lines.append(f"  - {dup.similarity_explanation}")
                    lines.append(f"  - {dup.recommendation}\n")

            if rev['duplicates']['codebase']:
                lines.append("### 💡 Better Implementation in Codebase\n")
                for dup in rev['duplicates']['codebase']:
                    lines.append(f"- **{dup.function_name}** in {dup.file_path}")
                    lines.append(f"  - {dup.similarity_explanation}")
                    lines.append(f"  - {dup.recommendation}\n")

            if any(rev['risks'].values()):
                lines.append("### 🔴 Risks\n")
                for risk_type, issues in rev['risks'].items():
                    if issues:
                        lines.append(f"**{risk_type.title()}:**")
                        for issue in issues:
                            lines.append(f"- {issue}")
                        lines.append("")

            lines.append("---\n")

        return "\n".join(lines)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="uncommitted")
    parser.add_argument("--base", default="main")
    args = parser.parse_args()

    config = ReviewConfig(
        comparison_mode=args.mode,
        base_branch=args.base
    )

    reviewer = DiffReviewer(config)
    reviews = reviewer.review_all_changes()

    if reviews:
        report = reviewer.generate_report(reviews)
        print("\n" + "="*80)
        print(report)

        with open("llm_review.md", "w") as f:
            f.write(report)
        print("\n✅ Saved to llm_review.md")


if __name__ == "__main__":
    main()