from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.prompts import PromptTemplate
from config.config import llm_caller


# ==========================================================
# STRUCTURED OUTPUT
# ==========================================================

class DomainClassificationOutput(BaseModel):

    domain: Literal[
        "HEALTHCARE",
        "ENGINEERING"
    ] = Field(
        description="Predicted domain."
    )

    confidence: float = Field(
        description="Confidence score between 0.0 and 1.0."
    )

    reason: str = Field(
        description="Reason for selecting the domain."
    )


# ==========================================================
# DOMAIN CLASSIFIER
# ==========================================================

def classify_domain(state_partial):
    """
    Classifies a query into either:

    - HEALTHCARE
    - ENGINEERING

    Returns:
    {
        domain,
        confidence,
        reason
    }
    """

    domain_prompt = PromptTemplate.from_template(
        """
You are an expert domain classifier.

You MUST classify the text into exactly one domain.

AVAILABLE DOMAINS

1. HEALTHCARE
   Examples:
   - Diseases
   - Treatments
   - Hospitals
   - Patients
   - Doctors
   - Medicine
   - Medical diagnosis
   - Clinical reports
   - Healthcare systems

2. ENGINEERING
   Examples:
   - Software development
   - Programming
   - Artificial Intelligence
   - Machine Learning
   - Networking
   - Mechanical engineering
   - Electrical engineering
   - Civil engineering
   - System design

RULES

- Choose ONLY one domain.
- Do not invent new domains.
- Return confidence between 0 and 1.
- Explain why the selected domain is most appropriate.

INPUT TEXT

{snippet_text}
"""
    )

    llm = llm_caller()

    structured_llm = llm.with_structured_output(
        DomainClassificationOutput
    )

    classification_chain = (
        domain_prompt
        | structured_llm
    )

    result = classification_chain.invoke(
        state_partial
    )

    return result
