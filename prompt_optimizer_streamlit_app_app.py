import textwrap
import json
from datetime import datetime
import streamlit as st

# ---------------------------
# App Config
# ---------------------------
st.set_page_config(
    page_title="Prompt Optimizer · Streamlit",
    page_icon="✨",
    layout="wide",
)

st.title("✨ Prompt Optimizer for ChatGPT")
st.caption("Turn a rough idea into a crisp, production-ready prompt engineering spec.")

# ---------------------------
# Helper Functions
# ---------------------------

def sanitize(text: str) -> str:
    return (text or "").strip()


def as_json_schema(example_keys: list[str]) -> str:
    """Return a minimal JSON schema for the chosen fields."""
    props = {k: {"type": "string"} for k in example_keys}
    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": props,
        "required": example_keys,
        "additionalProperties": False,
    }
    return json.dumps(schema, indent=2)


def default_style_guide():
    return (
        "Write clearly and concretely. Prefer active voice. Avoid hedging. "
        "Cite facts only when sources are provided. Use numbered steps when helpful."
    )


def build_optimized_prompt(
    raw_prompt: str,
    *,
    persona: str,
    audience: str,
    goal: str,
    tone: str,
    format_pref: str,
    length_pref: str,
    reading_level: str,
    must_haves: list[str],
    must_not: list[str],
    ask_clarifying: bool,
    include_quality_checklist: bool,
    include_self_eval: bool,
    output_json: bool,
    json_fields: list[str],
    few_shot: str,
    variables: dict,
    structure: str,
    lang: str,
) -> str:
    """Convert inputs into a structured, model-ready instruction set."""
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    sys_lines = []
    sys_lines.append(f"You are {persona or 'an expert assistant'}. Focus on {goal.lower() if goal else 'the task' } for the specified audience.")
    sys_lines.append("Follow the style guide and constraints exactly. Think privately; return only the final answer, not your chain-of-thought.")
    if tone:
        sys_lines.append(f"Maintain a {tone.lower()} tone appropriate for {audience or 'the user'}.")
    if reading_level:
        sys_lines.append(f"Target reading level: {reading_level}.")
    if structure:
        sys_lines.append(f"Structure: {structure}.")
    if lang and lang != "Auto":
        sys_lines.append(f"Respond in {lang}.")

    constraints = []
    if format_pref:
        constraints.append(f"Preferred output format: {format_pref}.")
    if length_pref:
        constraints.append(f"Conciseness/length: {length_pref}.")
    if must_haves:
        constraints.append("Must include: " + "; ".join(must_haves) + ".")
    if must_not:
        constraints.append("Do NOT include: " + "; ".join(must_not) + ".")
    constraints.append(default_style_guide())
    constraints.append("If information is missing or ambiguous, " + ("ask targeted clarifying questions first." if ask_clarifying else "state assumptions explicitly and proceed."))

    # Few-shot block
    few_shot_block = ""
    fs = sanitize(few_shot)
    if fs:
        few_shot_block = textwrap.dedent(f"""
        ### Few-shot Examples
        {fs}
        """)

    # Variables block (for templating workflows)
    variables_block = ""
    if variables:
        # Render a simple YAML-like variables section
        items = "\n".join([f"  {k}: {v}" for k, v in variables.items() if sanitize(v)])
        if items.strip():
            variables_block = textwrap.dedent(f"""
            ### Variables
            Use the following variables if referenced in the task. If not provided, ask for them or make reasonable defaults.
            ```yaml
            variables:
            {items}
            ```
            """)

    # Output JSON schema if requested
    json_block = ""
    if output_json and json_fields:
        json_block = textwrap.dedent(f"""
        ### Output JSON Schema
        The response must be valid JSON conforming to this schema:
        ```json
        {as_json_schema(json_fields)}
        ```
        Return only JSON in the final answer (no prose outside the JSON object).
        """)

    # Quality checklist
    quality_block = ""
    if include_quality_checklist:
        quality_block = textwrap.dedent(
            """
            ### Quality Checklist (before finalizing)
            - [ ] Directly addresses the stated goal and audience
            - [ ] Concrete, specific, and unambiguous
            - [ ] Contains all required elements and respects constraints
            - [ ] Correct tone and structure
            - [ ] No chain-of-thought or private reasoning exposed
            - [ ] If assumptions were made, they are listed at the end
            """
        )

    # Self-evaluation rubric
    self_eval_block = ""
    if include_self_eval:
        self_eval_block = textwrap.dedent(
            """
            ### Self-Evaluation (after drafting)
            Provide a brief, 3-bullet justification of why this response meets the goal and constraints. Keep it under 80 words total.
            """
        )

    # Build the final prompt
    optimized = textwrap.dedent(f"""
    ## System
    - Role: {persona or 'Expert Assistant'}
    - Date: {now}

    ## Instructions
    {"\n".join(sys_lines)}

    ### Constraints & Preferences
    {"\n".join(f'- {c}' for c in constraints)}

    ### Task (from user)
    {sanitize(raw_prompt)}

    ### Audience
    {sanitize(audience) or 'General'}

    {few_shot_block}{variables_block}{json_block}{quality_block}{self_eval_block}
    """)

    # Clean blank lines
    optimized = "\n".join([ln.rstrip() for ln in optimized.splitlines() if ln.strip() != ""]) + "\n"
    return optimized


def apply_transformations(raw_prompt: str) -> list[str]:
    """Heuristic suggestions shown to the user about how we improved the prompt."""
    ideas = []
    if len(raw_prompt.split()) < 6:
        ideas.append("Expanded a very short prompt into a detailed task with constraints and audience.")
    if any(w in raw_prompt.lower() for w in ["optimize", "improve", "rewrite"]):
        ideas.append("Preserved intent while clarifying success criteria and output format.")
    if not any(ch in raw_prompt for ch in ["?", "."]):
        ideas.append("Added clear instructions and removed ambiguity about expected deliverables.")
    ideas.append("Added optional JSON schema enforcement when selected.")
    ideas.append("Included a quality checklist and self-evaluation to raise answer quality.")
    return ideas

# ---------------------------
# Sidebar Controls
# ---------------------------
with st.sidebar:
    st.header("Settings")

    persona = st.text_input(
        "Assistant persona / role",
        value="a senior prompt engineer and domain expert",
        help="How the model should behave (e.g., 'AP English teacher', 'FAANG data scientist').",
    )

    goal = st.selectbox(
        "Primary goal",
        options=[
            "Explain",
            "Summarize",
            "Generate",
            "Rewrite",
            "Plan",
            "Classify",
            "Extract",
            "Code",
            "Debug",
            "Answer Q&A",
        ],
        index=2,
    )

    audience = st.text_input(
        "Target audience",
        value="startup founder with basic technical literacy",
    )

    tone = st.selectbox(
        "Tone",
        ["Neutral", "Professional", "Friendly", "Concise", "Academic", "Persuasive", "Playful"],
        index=1,
    )

    format_pref = st.selectbox(
        "Output format",
        ["Paragraphs", "Bulleted list", "Numbered steps", "Table", "JSON"],
        index=2,
    )

    length_pref = st.selectbox(
        "Length preference",
        ["Concise", "Medium", "Detailed"],
        index=1,
    )

    reading_level = st.selectbox(
        "Reading level",
        ["General", "Middle school", "High school", "Undergrad", "Graduate"],
        index=3,
    )

    structure = st.text_input(
        "Structure (optional)",
        value="Introduction ▸ Key Points ▸ Examples ▸ Actionable Steps",
    )

    lang = st.selectbox("Response language", ["Auto", "English", "Spanish", "French", "German", "Indonesian", "Thai", "Chinese"], index=0)

    st.markdown("---")

    must_haves = st.text_area(
        "Must include (one per line)",
        value="Actionable steps\nAssumptions\nReferences section when sources are given",
        height=96,
    ).splitlines()

    must_not = st.text_area(
        "Do NOT include (one per line)",
        value="Apologies\nSpeculation without evidence\nChain-of-thought",
        height=72,
    ).splitlines()

    ask_clarifying = st.checkbox("Ask clarifying questions if needed", value=True)
    include_quality_checklist = st.checkbox("Embed quality checklist", value=True)
    include_self_eval = st.checkbox("Embed self-evaluation rubric", value=False)

    st.markdown("---")
    output_json = st.checkbox("Enforce JSON output", value=False)
    json_fields_text = st.text_input(
        "JSON fields (comma-separated)",
        value="title, summary, steps",
    )
    json_fields = [f.strip() for f in json_fields_text.split(",") if f.strip()]

    st.markdown("---")
    st.subheader("Few-shot examples (optional)")
    st.caption("Paste pairs like: USER → ... / ASSISTANT → ... (multiple examples allowed)")
    few_shot = st.text_area("Few-shot block", height=140)

    st.subheader("Variables (optional)")
    st.caption("Define placeholders like {product}, {audience}. One per line as key: value")
    variables_input = st.text_area("Variables", value="product: Acme Analytics\naudience: growth PMs", height=100)
    variables = {}
    for line in variables_input.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            variables[sanitize(k)] = sanitize(v)

# ---------------------------
# Main Area
# ---------------------------
left, right = st.columns([1, 1])

with left:
    st.subheader("1) Your raw prompt or idea")
    raw_prompt = st.text_area(
        "Paste your initial prompt/idea",
        height=220,
        placeholder="e.g., 'Write a product spec for a new onboarding flow for a fintech app.'",
    )

    generate = st.button("Optimize Prompt ✨", type="primary")

with right:
    st.subheader("2) Optimized Prompt Engineering Spec")
    placeholder = st.empty()

if generate:
    optimized = build_optimized_prompt(
        raw_prompt or "Summarize the key points of this article for a busy executive.",
        persona=persona,
        audience=audience,
        goal=goal,
        tone=tone,
        format_pref=format_pref,
        length_pref=length_pref,
        reading_level=reading_level,
        must_haves=[m for m in must_haves if sanitize(m)],
        must_not=[m for m in must_not if sanitize(m)],
        ask_clarifying=ask_clarifying,
        include_quality_checklist=include_quality_checklist,
        include_self_eval=include_self_eval,
        output_json=output_json,
        json_fields=json_fields,
        few_shot=few_shot,
        variables=variables,
        structure=structure,
        lang=lang,
    )

    with right:
        placeholder.code(optimized, language="markdown")
        st.download_button(
            label="Download Prompt as .txt",
            data=optimized.encode("utf-8"),
            file_name="optimized_prompt.txt",
            mime="text/plain",
        )

        with st.expander("What changed (optimizer notes)"):
            for idea in apply_transformations(raw_prompt):
                st.write("• ", idea)

# Footer tip
st.markdown(
    """
    <div style="margin-top:1rem; opacity:0.7; font-size:0.9em;">
    Tip: Save common settings in the sidebar, then paste any rough prompt—this app will emit a clean, structured spec ready for ChatGPT or API calls.
    </div>
    """,
    unsafe_allow_html=True,
)
