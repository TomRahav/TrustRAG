#!/usr/bin/env python3
"""
BEIR Standard Synthetic Dataset Generator using Gemini 2.5 Pro

This script uses Gemini 2.5 Pro to generate high-quality synthetic BEIR datasets with:
1. Corpus of synthetic documents
2. Queries answerable only from the corpus
3. Correct answers based on existing documents
4. Adversarial queries pointing to incorrect answers

Uses Google's Gemini API for high-quality content generation.
"""

import json
import random
import os
from typing import Dict, List, Optional
from pathlib import Path
import argparse
from datetime import datetime
import time
from dataclasses import dataclass
import re

# Gemini API imports
try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print(
        "❌ Google GenerativeAI not installed. Install with: pip install google-generativeai"
    )


@dataclass
class DocumentInfo:
    doc_id: str
    title: str
    text: str
    domain: str
    entities: Dict[str, List[str]]


@dataclass
class QueryInfo:
    query_id: str
    query_text: str
    answer: str
    source_docs: List[str]
    adversarial_queries: List[Dict[str, str]]


class GeminiGenerator:
    """Interface for Gemini 2.5 Pro text generation."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-1.5-pro",
    ):
        if not GEMINI_AVAILABLE:
            raise ImportError("Google GenerativeAI library not available")

        # Configure API key
        if api_key:
            genai.configure(api_key=api_key)
        else:
            # Try to get from environment
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError(
                    "Gemini API key required. Set GEMINI_API_KEY environment variable or pass api_key parameter.\n"
                    "Get your API key from: https://aistudio.google.com/app/apikey"
                )
            genai.configure(api_key=api_key)

        # Initialize model
        self.model_name = model_name
        self.model = genai.GenerativeModel(
            model_name=model_name,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            },
        )

        # Test the connection
        try:
            test_response = self.model.generate_content("Hello")
            print(f"✅ Gemini {model_name} connected successfully! {test_response}")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Gemini API: {e}")

    def generate(
        self, prompt: str, temperature: float = 0.7, max_output_tokens: int = 512
    ) -> str:
        """Generate text using Gemini 2.5 Pro."""
        try:
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                top_p=0.9,
                top_k=40,
            )

            response = self.model.generate_content(
                prompt, generation_config=generation_config
            )

            if response.candidates and response.candidates[0].content.parts:
                return response.candidates[0].content.parts[0].text
            else:
                print(f"⚠️ No content generated for prompt: {prompt[:100]}...")
                return ""

        except Exception as e:
            print(f"❌ Gemini generation failed: {e}")
            # Add a small delay to avoid rate limiting
            time.sleep(1)
            return ""


class BEIRSyntheticGenerator:
    def __init__(
        self,
        output_dir: str = "synthetic_beir_dataset",
        api_key: Optional[str] = None,
        model_name: str = "gemini-1.5-pro",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Initialize Gemini generator
        print(f"🤖 Initializing Gemini {model_name}...")
        self.llm = GeminiGenerator(api_key=api_key, model_name=model_name)

        # Dataset structures
        self.documents: List[DocumentInfo] = []
        self.queries: List[QueryInfo] = []

        # Domain configuration
        self.domains = [
            "financial_reports",
            "technical_specifications",
            "corporate_communications",
            "research_papers",
            "legal_documents",
            "product_manuals",
            "market_analysis",
            "scientific_studies",
            "business_contracts",
            "operational_procedures",
        ]

        # Rate limiting
        self.request_delay = 0.5  # Delay between requests to avoid rate limits

    def generate_dataset(
        self,
        num_docs: int = 1000,
        num_queries: int = 200,
        adversarial_per_query: int = 3,
    ):
        """Generate complete BEIR dataset."""
        print("🏗️ Generating BEIR synthetic dataset with Gemini 2.5 Pro...")
        print(f"   📄 Documents: {num_docs}")
        print(f"   ❓ Queries: {num_queries}")
        print(f"   ⚔️ Adversarial queries per query: {adversarial_per_query}")
        print(f"   ⏱️ Rate limit delay: {self.request_delay}s between requests")

        # Generate document corpus
        print("\n📚 Generating document corpus...")
        self._generate_document_corpus(num_docs)

        # Generate queries and answers
        print("❓ Generating queries and answers...")
        self._generate_queries_and_answers(num_queries, adversarial_per_query)

        # Save in BEIR format
        print("💾 Saving in BEIR format...")
        self._save_beir_format()

        print(f"\n✅ Dataset generated successfully in: {self.output_dir}")

    def _generate_document_corpus(self, num_docs: int):
        """Generate synthetic document corpus using Gemini 2.5 Pro."""
        docs_per_domain = num_docs // len(self.domains)

        for domain_idx, domain in enumerate(self.domains):
            print(f"  📝 Generating {docs_per_domain} documents for {domain}...")

            for i in range(docs_per_domain):
                doc = self._generate_single_document(domain, i)
                if doc:
                    self.documents.append(doc)

                # Rate limiting
                time.sleep(self.request_delay)

                # Progress indicator
                if (i + 1) % 20 == 0:
                    print(f"    Generated {i + 1}/{docs_per_domain} {domain} documents")

        print(f"📊 Total documents generated: {len(self.documents)}")

    def _generate_single_document(
        self, domain: str, doc_index: int
    ) -> Optional[DocumentInfo]:
        """Generate a single document using Gemini 2.5 Pro."""

        prompt = self._get_document_generation_prompt(domain)

        # Generate document
        generated_text = self.llm.generate(
            prompt, temperature=0.8, max_output_tokens=800
        )

        if not generated_text:
            return None

        # Parse the generated content
        doc_content = self._parse_document_content(generated_text)

        if not doc_content:
            return None

        doc_id = f"doc_{domain}_{doc_index:06d}"

        # Extract entities from the document
        entities = self._extract_entities_with_llm(doc_content["text"])

        return DocumentInfo(
            doc_id=doc_id,
            title=doc_content["title"],
            text=doc_content["text"],
            domain=domain,
            entities=entities,
        )

    def _get_document_generation_prompt(self, domain: str) -> str:
        """Get domain-specific document generation prompt optimized for Gemini."""

        prompts = {
            "financial_reports": """
Create a realistic financial report document with specific details. Use this exact format:

TITLE: [Specific company financial report title]
CONTENT: [Detailed financial report content]

Requirements:
- Include specific company names (fictional but realistic)
- Include concrete financial figures (revenue, profit, costs)
- Include executive names and titles
- Include specific percentages and growth metrics
- Include quarters/years (2020-2024)
- Include specific business deals and partnerships
- Make it detailed and realistic (400-600 words)

Example entities to include: Company names like "TechFlow Industries", "DataCore Systems", executive names like "Sarah Chen, CFO", specific amounts like "$125 million revenue", "18% growth", etc.
""",
            "technical_specifications": """
Create a detailed technical specification document. Use this exact format:

TITLE: [Specific product/system technical specification title]
CONTENT: [Detailed technical specification content]

Requirements:
- Include specific product/system names
- Include concrete performance metrics and benchmarks
- Include technical requirements and specifications
- Include version numbers and compatibility information
- Include testing results and measurements
- Include engineering team members and locations
- Make it detailed and realistic (400-600 words)

Example entities: Product names like "AlphaCore Processing Unit", "BetaMax Database System", performance metrics like "99.7% uptime", "2.3ms response time", engineer names like "Dr. Michael Torres, Lead Engineer", etc.
""",
            "corporate_communications": """
Create a realistic corporate communication document. Use this exact format:

TITLE: [Specific corporate communication title]
CONTENT: [Detailed corporate communication content]

Requirements:
- Include specific company names and leadership
- Include employee counts and organizational details
- Include business metrics and achievements
- Include partnership and acquisition information
- Include location and facility information
- Include specific dates and timelines
- Make it detailed and realistic (400-600 words)

Example entities: Company names like "InnovateTech Corporation", "Global Solutions Ltd", executives like "Jennifer Martinez, CEO", metrics like "15,000 employees", "45% market growth", locations like "Silicon Valley headquarters", etc.
""",
            "research_papers": """
Create a realistic research paper or study document. Use this exact format:

TITLE: [Specific research paper title]
CONTENT: [Detailed research content including methodology and findings]

Requirements:
- Include specific researcher names and affiliations
- Include concrete study methodologies and sample sizes
- Include specific results, percentages, and statistical data
- Include research timeframes and locations
- Include funding sources and collaborations
- Include conclusions and implications
- Make it detailed and realistic (400-600 words)

Example entities: Researcher names like "Dr. Elena Rodriguez, Stanford University", study details like "5,000 participants", "87% accuracy improvement", "12-month longitudinal study", funding like "NSF Grant #2024-1234", etc.
""",
            "legal_documents": """
Create a realistic legal document. Use this exact format:

TITLE: [Specific legal document title]
CONTENT: [Detailed legal document content]

Requirements:
- Include specific party names and legal entities
- Include concrete contract terms and financial amounts
- Include specific durations and deadlines
- Include legal compliance requirements
- Include jurisdiction and governing law information
- Include penalty clauses and obligations
- Make it detailed and realistic (400-600 words)

Example entities: Company names like "MegaCorp Industries Inc.", "StartupTech LLC", legal terms like "$50 million contract value", "3-year agreement term", "California jurisdiction", lawyer names like "Attorney John Davis", etc.
""",
        }

        # Default prompt for other domains
        default_prompt = f"""
Create a realistic {domain.replace('_', ' ')} document with specific details. Use this exact format:

TITLE: [Specific document title for {domain.replace('_', ' ')}]
CONTENT: [Detailed content with specific information]

Requirements:
- Include specific names, companies, and people
- Include concrete numbers, dates, and metrics
- Include relevant technical or business details
- Include locations and organizational information
- Make it detailed and realistic (400-600 words)
- Use specific entities relevant to {domain.replace('_', ' ')}

Create content that sounds professional and contains factual information that could be used to answer specific questions.
"""

        return prompts.get(domain, default_prompt)

    def _parse_document_content(self, generated_text: str) -> Optional[Dict[str, str]]:
        """Parse the generated document content."""
        try:
            # Clean up the generated text
            text = generated_text.strip()

            # Look for TITLE: and CONTENT: markers
            title_match = re.search(
                r"TITLE:\s*(.+?)(?=\n|CONTENT:)", text, re.IGNORECASE
            )
            content_match = re.search(
                r"CONTENT:\s*(.*)", text, re.IGNORECASE | re.DOTALL
            )

            if title_match and content_match:
                title = title_match.group(1).strip()
                content = content_match.group(1).strip()
            else:
                # Fallback: use first line as title, rest as content
                lines = text.split("\n")
                title = lines[0][:100] + "..." if len(lines[0]) > 100 else lines[0]
                content = "\n".join(lines[1:]) if len(lines) > 1 else text

            # Clean up title and content
            title = re.sub(r"^[#*\-\s]*", "", title).strip()
            content = re.sub(
                r"^\s*CONTENT:\s*", "", content, flags=re.IGNORECASE
            ).strip()

            # Validate content length
            if len(content) < 200:  # Filter out too short content
                return None

            return {"title": title, "text": content}

        except Exception as e:
            print(f"⚠️ Error parsing document: {e}")
            return None

    def _extract_entities_with_llm(self, text: str) -> Dict[str, List[str]]:
        """Extract entities from text using Gemini 2.5 Pro."""

        entity_prompt = f"""
Extract specific entities from the following text. Return ONLY the entities in this exact format:

COMPANIES: [list company names separated by commas, or "None"]
PEOPLE: [list person names separated by commas, or "None"]  
PRODUCTS: [list product/service names separated by commas, or "None"]
NUMBERS: [list important financial numbers/percentages separated by commas, or "None"]
LOCATIONS: [list locations/places separated by commas, or "None"]

Text to analyze:
{text[:1000]}

Important:
- Extract only entities explicitly mentioned in the text
- Use the exact format shown above
- If no entities of a type are found, write "None"
- For numbers, include currency amounts, percentages, and key metrics
- Be specific and accurate
"""

        entity_response = self.llm.generate(
            entity_prompt, temperature=0.3, max_output_tokens=300
        )

        # Add delay for rate limiting
        time.sleep(self.request_delay)

        # Parse entity response
        entities = {
            "companies": [],
            "people": [],
            "products": [],
            "numbers": [],
            "locations": [],
        }

        try:
            lines = entity_response.split("\n")
            for line in lines:
                line = line.strip()
                if line.startswith("COMPANIES:"):
                    entities["companies"] = self._parse_entity_list(
                        line.replace("COMPANIES:", "")
                    )
                elif line.startswith("PEOPLE:"):
                    entities["people"] = self._parse_entity_list(
                        line.replace("PEOPLE:", "")
                    )
                elif line.startswith("PRODUCTS:"):
                    entities["products"] = self._parse_entity_list(
                        line.replace("PRODUCTS:", "")
                    )
                elif line.startswith("NUMBERS:"):
                    entities["numbers"] = self._parse_entity_list(
                        line.replace("NUMBERS:", "")
                    )
                elif line.startswith("LOCATIONS:"):
                    entities["locations"] = self._parse_entity_list(
                        line.replace("LOCATIONS:", "")
                    )

        except Exception as e:
            print(f"⚠️ Error parsing entities: {e}")

        return entities

    def _parse_entity_list(self, entity_string: str) -> List[str]:
        """Parse comma-separated entity list."""
        if not entity_string or entity_string.strip().lower() in [
            "none",
            "n/a",
            "not applicable",
        ]:
            return []

        entities = [e.strip() for e in entity_string.split(",")]
        return [
            e
            for e in entities
            if e and e.lower() not in ["none", "n/a", "not applicable"]
        ]

    def _generate_queries_and_answers(
        self, num_queries: int, adversarial_per_query: int
    ):
        """Generate queries and answers using Gemini 2.5 Pro."""

        for i in range(num_queries):
            # Select random documents for this query
            selected_docs = random.sample(
                self.documents, random.randint(1, min(3, len(self.documents)))
            )

            query_info = self._generate_single_query(
                selected_docs, i, adversarial_per_query
            )

            if query_info:
                self.queries.append(query_info)

            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"    Generated {i + 1}/{num_queries} queries")

        print(f"📊 Total queries generated: {len(self.queries)}")

    def _generate_single_query(
        self, docs: List[DocumentInfo], query_index: int, adversarial_count: int
    ) -> Optional[QueryInfo]:
        """Generate a single query with answer and adversarial queries."""

        # Combine document content for context
        context = ""
        for doc in docs:
            context += f"Document ID: {doc.doc_id}\nTitle: {doc.title}\nContent: {doc.text[:600]}...\n\n"

        query_prompt = f"""
Based on the following documents, create a specific factual question that can ONLY be answered using information from these documents. The question should require finding specific facts, numbers, names, or relationships mentioned in the documents.

{context}

Create a question-answer pair using this exact format:

QUESTION: [A specific, factual question about information in the documents]
ANSWER: [The precise answer found in the documents]

Requirements:
- The question must be answerable only from the provided documents
- Ask for specific facts: names, numbers, dates, amounts, relationships
- The answer should be concrete and specific
- Avoid general or opinion-based questions
- Make the question challenging but fair

Examples of good questions:
- "What was the cost of the deal with [Company X]?"
- "Who is the CEO of [Company Y]?"
- "What percentage improvement was reported in [specific metric]?"
- "When did [specific event] occur?"
"""

        query_response = self.llm.generate(
            query_prompt, temperature=0.7, max_output_tokens=200
        )

        # Add delay for rate limiting
        time.sleep(self.request_delay)

        # Parse query and answer
        query_data = self._parse_query_response(query_response)
        if not query_data:
            return None

        # Generate adversarial queries
        adversarial_queries = []
        for j in range(adversarial_count):
            adversarial = self._generate_adversarial_query(
                query_data["question"], query_data["answer"], docs
            )
            if adversarial:
                adversarial_queries.append(adversarial)

            # Small delay between adversarial generations
            time.sleep(self.request_delay * 0.5)

        query_id = f"query_{query_index:06d}"
        source_doc_ids = [doc.doc_id for doc in docs]

        return QueryInfo(
            query_id=query_id,
            query_text=query_data["question"],
            answer=query_data["answer"],
            source_docs=source_doc_ids,
            adversarial_queries=adversarial_queries,
        )

    def _parse_query_response(self, response: str) -> Optional[Dict[str, str]]:
        """Parse the query generation response."""
        try:
            text = response.strip()

            # Look for QUESTION: and ANSWER: patterns
            question_match = re.search(
                r"QUESTION:\s*(.+?)(?=\n|ANSWER:)", text, re.IGNORECASE
            )
            answer_match = re.search(
                r"ANSWER:\s*(.+?)(?=\n|$)", text, re.IGNORECASE | re.DOTALL
            )

            if question_match and answer_match:
                question = question_match.group(1).strip()
                answer = answer_match.group(1).strip()

                # Clean up
                question = re.sub(r"^[#*\-\s]*", "", question).strip()
                answer = re.sub(r"^[#*\-\s]*", "", answer).strip()

                if question and answer:
                    return {"question": question, "answer": answer}

            return None

        except Exception as e:
            print(f"⚠️ Error parsing query response: {e}")
            return None

    def _generate_adversarial_query(
        self, original_query: str, correct_answer: str, docs: List[DocumentInfo]
    ) -> Optional[Dict[str, str]]:
        """Generate an adversarial query using Gemini 2.5 Pro."""

        adversarial_prompt = f"""
Create an adversarial question based on the original question below. The adversarial question should be designed to mislead while appearing similar to the original.

Original Question: {original_query}
Correct Answer: {correct_answer}

Create an adversarial version using this exact format:

ADVERSARIAL_QUESTION: [A misleading version of the question that looks similar but asks for different information]
INCORRECT_ANSWER: [What someone might incorrectly answer if they don't read carefully]
STRATEGY: [Brief description of how this question is misleading]

Adversarial strategies to use:
- Entity substitution (change names/companies)
- Temporal confusion (different time periods)
- Numerical confusion (different metrics)
- Relationship reversal (who did what to whom)
- Context switching (similar but different scenarios)

Make the adversarial question realistic and tricky but not obviously wrong.
"""

        adversarial_response = self.llm.generate(
            adversarial_prompt, temperature=0.8, max_output_tokens=250
        )

        # Parse adversarial response
        try:
            text = adversarial_response.strip()

            adv_question_match = re.search(
                r"ADVERSARIAL_QUESTION:\s*(.+?)(?=\n|INCORRECT_ANSWER:)",
                text,
                re.IGNORECASE,
            )
            incorrect_answer_match = re.search(
                r"INCORRECT_ANSWER:\s*(.+?)(?=\n|STRATEGY:)", text, re.IGNORECASE
            )
            strategy_match = re.search(
                r"STRATEGY:\s*(.+?)(?=\n|$)", text, re.IGNORECASE | re.DOTALL
            )

            if adv_question_match and incorrect_answer_match:
                adversarial_question = adv_question_match.group(1).strip()
                incorrect_answer = incorrect_answer_match.group(1).strip()
                strategy = (
                    strategy_match.group(1).strip() if strategy_match else "adversarial"
                )

                # Clean up
                adversarial_question = re.sub(
                    r"^[#*\-\s]*", "", adversarial_question
                ).strip()
                incorrect_answer = re.sub(r"^[#*\-\s]*", "", incorrect_answer).strip()

                if adversarial_question and incorrect_answer:
                    return {
                        "query": adversarial_question,
                        "correct_answer": correct_answer,
                        "adversarial_answer": incorrect_answer,
                        "strategy": strategy,
                    }

            return None

        except Exception as e:
            print(f"⚠️ Error parsing adversarial response: {e}")
            return None

    def _save_beir_format(self):
        """Save dataset in BEIR format."""

        # Create corpus.jsonl
        corpus_file = self.output_dir / "corpus.jsonl"
        with open(corpus_file, "w", encoding="utf-8") as f:
            for doc in self.documents:
                corpus_entry = {
                    "_id": doc.doc_id,
                    "title": doc.title,
                    "text": doc.text,
                    "metadata": {"domain": doc.domain, "entities": doc.entities},
                }
                f.write(json.dumps(corpus_entry, ensure_ascii=False) + "\n")

        # Create queries.jsonl
        queries_file = self.output_dir / "queries.jsonl"
        with open(queries_file, "w", encoding="utf-8") as f:
            for query in self.queries:
                query_entry = {
                    "_id": query.query_id,
                    "text": query.query_text,
                    "metadata": {
                        "answer": query.answer,
                        "source_docs": query.source_docs,
                    },
                }
                f.write(json.dumps(query_entry, ensure_ascii=False) + "\n")

        # Create qrels.tsv (query relevance judgments)
        qrels_file = self.output_dir / "qrels.tsv"
        with open(qrels_file, "w", encoding="utf-8") as f:
            f.write("query-id\tcorpus-id\tscore\n")
            for query in self.queries:
                for doc_id in query.source_docs:
                    f.write(f"{query.query_id}\t{doc_id}\t1\n")

        # Create adversarial_queries.jsonl
        adversarial_file = self.output_dir / "adversarial_queries.jsonl"
        with open(adversarial_file, "w", encoding="utf-8") as f:
            for query in self.queries:
                adversarial_entry = {
                    "original_query_id": query.query_id,
                    "original_query": query.query_text,
                    "correct_answer": query.answer,
                    "adversarial_queries": query.adversarial_queries,
                }
                f.write(json.dumps(adversarial_entry, ensure_ascii=False) + "\n")

        # Create dataset info
        info = {
            "dataset_name": "Synthetic BEIR Dataset (Gemini 2.5 Pro)",
            "generated_at": datetime.now().isoformat(),
            "model": "Gemini 2.5 Pro",
            "statistics": {
                "num_documents": len(self.documents),
                "num_queries": len(self.queries),
                "domains": list(set(doc.domain for doc in self.documents)),
                "avg_doc_length": sum(len(doc.text.split()) for doc in self.documents)
                // max(len(self.documents), 1),
                "avg_query_length": sum(
                    len(query.query_text.split()) for query in self.queries
                )
                // max(len(self.queries), 1),
                "total_adversarial_queries": sum(
                    len(query.adversarial_queries) for query in self.queries
                ),
            },
        }

        with open(self.output_dir / "dataset_info.json", "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)

        print("📁 Files saved:")
        print(f"   📄 corpus.jsonl ({len(self.documents)} documents)")
        print(f"   ❓ queries.jsonl ({len(self.queries)} queries)")
        print("   🔗 qrels.tsv (relevance judgments)")
        print("   ⚔️ adversarial_queries.jsonl (adversarial examples)")
        print("   ℹ️ dataset_info.json (metadata)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate BEIR synthetic dataset using Gemini 2.5 Pro"
    )
    parser.add_argument(
        "--docs", type=int, default=10, help="Number of documents to generate"
    )
    parser.add_argument(
        "--queries", type=int, default=5, help="Number of queries to generate"
    )
    parser.add_argument(
        "--adversarial", type=int, default=1, help="Adversarial queries per main query"
    )
    parser.add_argument(
        "--api-key", help="Gemini API key (or set GEMINI_API_KEY env var)"
    )
    parser.add_argument("--model", default="gemini-1.5-pro", help="Gemini model name")
    parser.add_argument(
        "--output", default="synthetic_beir_dataset", help="Output directory"
    )
    parser.add_argument(
        "--delay", type=float, default=0.5, help="Delay between API requests (seconds)"
    )

    args = parser.parse_args()

    print("🤖 BEIR Synthetic Dataset Generator with Gemini 2.5 Pro")
    print("=" * 55)

    if not GEMINI_AVAILABLE:
        print("❌ Error: google-generativeai not installed")
        print("Install with: pip install google-generativeai")
        return 1

    if (
        not args.api_key
        and not os.getenv("GEMINI_API_KEY")
        and not os.getenv("GOOGLE_API_KEY")
    ):
        print("❌ Error: Gemini API key required")
        print("Set GEMINI_API_KEY environment variable or use --api-key")
        print("Get your API key from: https://aistudio.google.com/app/apikey")
        return 1

    try:
        generator = BEIRSyntheticGenerator(
            output_dir=args.output, api_key=args.api_key, model_name=args.model
        )

        # Set rate limiting delay
        generator.request_delay = args.delay

        generator.generate_dataset(
            num_docs=args.docs,
            num_queries=args.queries,
            adversarial_per_query=args.adversarial,
        )

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
