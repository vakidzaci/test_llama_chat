"""
Upload Python code to ChromaDB with proper metadata
Usage: python upload_to_chromadb.py --path ./rpa_bot --model py-embed --collection bot --persist ./chroma_db
"""
import argparse
import re
from pathlib import Path
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


class CodeMetadataExtractor:
    """Extract metadata from Python code"""

    @staticmethod
    def classify_file(filename: str, content: str) -> str:
        """Classify file type"""
        filename_lower = filename.lower()

        if filename_lower == "robot.py":
            return "robot"
        elif filename_lower == "steps.py":
            return "steps"
        elif filename_lower in ["utils.py", "util.py"]:
            return "utils"
        elif filename_lower == "config.py":
            return "config"
        elif filename_lower in ["model.py", "models.py"]:
            return "model"
        elif "test" in filename_lower:
            return "test"
        else:
            return "other"

    @staticmethod
    def extract_pattern_types(content: str) -> List[str]:
        """Extract what patterns/techniques this code demonstrates"""
        patterns = []

        content_lower = content.lower()

        # Error handling patterns
        if "try:" in content and "except" in content:
            patterns.append("error_handling")
        if "raise" in content:
            patterns.append("exception_raising")

        # Logging
        if "logging" in content_lower or "logger" in content_lower:
            patterns.append("logging")

        # Database
        if "postgresql" in content_lower or "psycopg" in content_lower:
            patterns.append("postgresql")
        if "select" in content_lower or "insert" in content_lower:
            patterns.append("sql")
        if "session" in content_lower and "commit" in content_lower:
            patterns.append("db_transaction")

        # Retry logic
        if "retry" in content_lower or "retries" in content_lower:
            patterns.append("retry_logic")

        # Validation
        if "validate" in content_lower or "assert" in content:
            patterns.append("validation")

        # Celery
        if "@task" in content or "celery" in content_lower:
            patterns.append("celery_task")

        # Security
        if "password" in content_lower or "credentials" in content_lower:
            patterns.append("security")
        if "encrypt" in content_lower or "decrypt" in content_lower:
            patterns.append("encryption")

        # File operations
        if "open(" in content or "with open" in content:
            patterns.append("file_io")

        # Web/API
        if "requests." in content or "selenium" in content_lower:
            patterns.append("web_automation")
        if "beautifulsoup" in content_lower or "soup" in content_lower:
            patterns.append("web_scraping")

        # Excel
        if "openpyxl" in content_lower or "pandas" in content_lower:
            patterns.append("excel")

        return patterns if patterns else ["general"]

    @staticmethod
    def extract_functions_and_classes(content: str) -> Dict[int, Dict]:
        """Extract function and class definitions with line numbers"""
        elements = {}
        lines = content.split('\n')

        for i, line in enumerate(lines, 1):
            stripped = line.lstrip()

            # Function definition
            func_match = re.match(r'^def\s+(\w+)\s*\(', stripped)
            if func_match:
                elements[i] = {
                    'type': 'function',
                    'name': func_match.group(1),
                    'indent': len(line) - len(stripped)
                }

            # Class definition
            class_match = re.match(r'^class\s+(\w+)', stripped)
            if class_match:
                elements[i] = {
                    'type': 'class',
                    'name': class_match.group(1),
                    'indent': len(line) - len(stripped)
                }

        return elements

    @staticmethod
    def get_containing_element(line_num: int, elements: Dict[int, Dict]) -> Optional[Dict]:
        """Find which function/class contains this line"""
        current_element = None

        for elem_line in sorted(elements.keys()):
            if elem_line <= line_num:
                current_element = {
                    'line': elem_line,
                    **elements[elem_line]
                }
            else:
                break

        return current_element

    @staticmethod
    def detect_quality_indicators(content: str) -> str:
        """Detect code quality level"""
        score = 0

        # Positive indicators
        if '"""' in content or "'''" in content:  # Docstrings
            score += 2
        if "try:" in content and "except" in content:  # Error handling
            score += 2
        if "logging" in content.lower():  # Logging
            score += 1
        if re.search(r'def \w+\([^)]*\) -> ', content):  # Type hints
            score += 2
        if "raise" in content:  # Explicit exceptions
            score += 1

        # Negative indicators
        if "pass" in content and content.count("pass") > 2:  # Too many pass statements
            score -= 1
        if re.search(r'except:(?!\s*\n\s*raise)', content):  # Bare except
            score -= 2

        if score >= 5:
            return "best_practice"
        elif score >= 2:
            return "good"
        else:
            return "acceptable"


class CodeUploader:
    """Upload code to ChromaDB with metadata"""

    def __init__(self,
                 path_to_folder: str,
                 model_ollama_name: str,
                 collection_name: str,
                 persist_dir: str,
                 ollama_host: str = "http://localhost:11434"):

        self.folder_path = Path(path_to_folder)
        self.collection_name = collection_name
        self.persist_dir = Path(persist_dir)

        # Initialize embeddings
        print(f"🔧 Initializing Ollama embeddings: {model_ollama_name}")
        self.embeddings = OllamaEmbeddings(
            model=model_ollama_name,
            base_url=ollama_host
        )

        # Initialize ChromaDB with persistence
        print(f"🗄️  Initializing ChromaDB at: {persist_dir}")
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir)
        )

        # Create or get collection
        self.collection = self._setup_collection()

        # Text splitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\nclass ", "\n\ndef ", "\n\n", "\n", " ", ""]
        )

        self.metadata_extractor = CodeMetadataExtractor()

    def _setup_collection(self):
        """Setup ChromaDB collection"""
        try:
            # Try to get existing collection
            collection = self.client.get_collection(self.collection_name)
            print(f"📚 Found existing collection: '{self.collection_name}' with {collection.count()} items")

            response = input("⚠️  Collection exists. Delete and recreate? (y/n): ")
            if response.lower() == 'y':
                self.client.delete_collection(self.collection_name)
                print(f"🗑️  Deleted collection: '{self.collection_name}'")
                collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": f"Code collection for {self.collection_name}"}
                )
                print(f"✨ Created new collection: '{self.collection_name}'")
        except:
            # Collection doesn't exist, create it
            collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": f"Code collection for {self.collection_name}"}
            )
            print(f"✨ Created new collection: '{self.collection_name}'")

        return collection

    def upload(self):
        """Upload all Python files from folder"""
        python_files = list(self.folder_path.rglob("*.py"))

        if not python_files:
            print(f"⚠️  No Python files found in {self.folder_path}")
            return

        print(f"📂 Found {len(python_files)} Python files")
        print()

        total_chunks = 0

        for file_path in python_files:
            chunks_added = self._process_file(file_path)
            total_chunks += chunks_added

        print()
        print("=" * 80)
        print(f"✅ Upload complete!")
        print(f"📊 Total files processed: {len(python_files)}")
        print(f"📊 Total chunks added: {total_chunks}")
        print(f"📊 Collection size: {self.collection.count()}")
        print(f"💾 Persisted to: {self.persist_dir}")
        print("=" * 80)

    def _process_file(self, file_path: Path) -> int:
        """Process a single file and upload to ChromaDB"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"⚠️  Error reading {file_path.name}: {e}")
            return 0

        # Skip empty files
        if not content.strip():
            print(f"⏭️  Skipping empty file: {file_path.name}")
            return 0

        # Extract metadata
        filename = file_path.name
        file_type = self.metadata_extractor.classify_file(filename, content)
        pattern_types = self.metadata_extractor.extract_pattern_types(content)
        quality_level = self.metadata_extractor.detect_quality_indicators(content)
        elements = self.metadata_extractor.extract_functions_and_classes(content)

        # Split into chunks
        chunks = self.splitter.split_text(content)

        # Calculate line numbers for each chunk
        chunk_line_info = self._calculate_chunk_lines(content, chunks)

        # Prepare data for ChromaDB
        ids = []
        documents = []
        metadatas = []
        embeddings_list = []

        for i, chunk in enumerate(chunks):
            chunk_id = f"{self.collection_name}_{file_path.stem}_{i}"

            line_start, line_end = chunk_line_info[i]

            # Find containing function/class
            containing_element = self.metadata_extractor.get_containing_element(
                line_start, elements
            )

            metadata = {
                "filename": filename,
                "filepath": str(file_path),
                "type": file_type,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "line_start": line_start,
                "line_end": line_end,
                "pattern_types": ",".join(pattern_types),
                "quality_level": quality_level,
            }

            # Add function/class info if available
            if containing_element:
                metadata["element_type"] = containing_element['type']
                metadata["element_name"] = containing_element['name']

            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append(metadata)

        # Generate embeddings
        print(f"⚙️  Processing {filename} ({len(chunks)} chunks)...", end=" ")
        embeddings_list = self.embeddings.embed_documents(documents)

        # Add to collection
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings_list
        )

        print(f"✅ Added {len(chunks)} chunks")

        return len(chunks)

    def _calculate_chunk_lines(self, content: str, chunks: List[str]) -> List[tuple]:
        """Calculate line numbers for each chunk"""
        lines = content.split('\n')
        total_lines = len(lines)

        chunk_lines = []
        current_line = 1

        for chunk in chunks:
            chunk_line_count = chunk.count('\n') + 1
            line_start = current_line
            line_end = min(current_line + chunk_line_count - 1, total_lines)

            chunk_lines.append((line_start, line_end))
            current_line = line_end + 1

        return chunk_lines


def main():
    parser = argparse.ArgumentParser(
        description="Upload Python code to ChromaDB with metadata"
    )
    parser.add_argument(
        "--path",
        required=True,
        help="Path to folder containing Python files"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Ollama embedding model name (e.g., py-embed)"
    )
    parser.add_argument(
        "--collection",
        required=True,
        help="ChromaDB collection name (e.g., bot, codebase)"
    )
    parser.add_argument(
        "--persist",
        required=True,
        help="Path to persist ChromaDB data (e.g., ./chroma_db)"
    )
    parser.add_argument(
        "--ollama-host",
        default="http://localhost:11434",
        help="Ollama server host (default: http://localhost:11434)"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("📤 CODE UPLOAD TO CHROMADB")
    print("=" * 80)
    print(f"📁 Source folder: {args.path}")
    print(f"🤖 Embedding model: {args.model}")
    print(f"📚 Collection name: {args.collection}")
    print(f"💾 Persist directory: {args.persist}")
    print(f"🔗 Ollama host: {args.ollama_host}")
    print("=" * 80)
    print()

    uploader = CodeUploader(
        path_to_folder=args.path,
        model_ollama_name=args.model,
        collection_name=args.collection,
        persist_dir=args.persist,
        ollama_host=args.ollama_host
    )

    uploader.upload()


if __name__ == "__main__":
    main()