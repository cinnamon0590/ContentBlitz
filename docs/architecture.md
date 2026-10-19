# Architecture

ContentBlitz is a single LangGraph `StateGraph` (14 nodes) built from six specialized agents, three
tool nodes, and three control nodes (input guardrail, quality gate, finalize). The UI, the notebook
and the tests all reach it through one facade, `ContentBlitz` (`src/core/workflow.py`).

## 1. Layered view

```mermaid
flowchart TB
    subgraph UI["Presentation"]
        ST[Streamlit app<br/>Chat · Dashboard · Research · Images · Export · Settings]
        NB[Notebook walkthrough]
        LGS[LangGraph Studio<br/>langgraph dev]
    end
    subgraph APP["Application"]
        F[ContentBlitz facade<br/>run · stream · history · rescore · update_artifact]
        G[LangGraph StateGraph<br/>+ checkpointer]
    end
    subgraph AG["Agents (src/agents)"]
        QH[Query Handler] --- RA[Deep Research] --- BW[SEO Blog Writer]
        LW[LinkedIn Writer] --- IG[Image Generator] --- CS[Content Strategist]
    end
    subgraph SV["Services (src/integrations) — via AgentDeps"]
        LLM[LLMService<br/>smart/fast tiers + fallbacks]
        SC[SearchChain<br/>SerpAPI → Tavily → Perplexity + TTL cache]
        IC[ImageChain<br/>gpt-image-1 → gpt-image-1-mini → Gemini 2.5 Flash Image → placeholder]
        MOD[Moderation]
    end
    subgraph UT["Pipeline utilities (src/utils)"]
        GR[guardrails] --- CO[content_optimization] --- QV[quality_validation]
        EX[export_tools] --- CA[cache] --- OB[observability / CostTracker]
    end
    ST & NB --> F --> G --> AG
    LGS --> G
    AG --> SV & UT
```

## 2. Graph (generated from the compiled graph)

```mermaid
graph TD;
    __start__([start]) --> input_guardrail;
    input_guardrail -.-> query_handler;
    input_guardrail -.->|blocked| finalize;
    query_handler -.-> research_agent;
    query_handler -.-> blog_writer;
    query_handler -.-> linkedin_writer;
    query_handler -.-> image_generator;
    query_handler -.-> content_strategist;
    query_handler -.-> general;
    query_handler -.->|empty campaign| finalize;
    research_agent -.->|tools| research_tools;
    research_tools --> research_agent;
    research_agent -.->|done| post_research_router;
    post_research_router -.-> blog_writer;
    post_research_router -.-> linkedin_writer;
    post_research_router -.-> image_generator;
    post_research_router -.-> content_strategist;
    blog_writer -.->|tools| blog_tools;
    blog_tools --> blog_writer;
    blog_writer -.->|done| quality_gate;
    linkedin_writer -.->|tools| linkedin_tools;
    linkedin_tools --> linkedin_writer;
    linkedin_writer -.->|done| quality_gate;
    content_strategist --> quality_gate;
    quality_gate -.->|revise| blog_writer;
    quality_gate -.->|revise| linkedin_writer;
    quality_gate -.->|revise| content_strategist;
    quality_gate -.->|next campaign format| image_generator;
    quality_gate -.-> finalize;
    image_generator -.->|next campaign format| blog_writer;
    image_generator -.-> linkedin_writer;
    image_generator -.-> finalize;
    general --> finalize;
    finalize --> __end__([end]);
```

Regenerate the graph at any time:

```python
from src.core.workflow import ContentBlitz
print(ContentBlitz().graph.get_graph().draw_mermaid())
```

## 3. Research → blog sequence

```mermaid
sequenceDiagram
    autonumber
    actor U as User
    participant IG as input_guardrail
    participant QH as Query Handler
    participant RA as Research Agent
    participant SC as SearchChain
    participant BW as Blog Writer
    participant QG as Quality Gate
    participant FN as finalize
    U->>IG: "Write an SEO blog about email marketing for bakeries"
    IG->>IG: injection/policy screen · PII redaction · moderation
    IG->>QH: sanitised message (+ HumanMessage to memory)
    QH->>QH: structured ContentRequest (intent=blog, needs_research)
    QH->>RA: route=research
    loop tool loop (≤ 6)
        RA->>SC: web_search(query) / deep_research (≤1)
        SC-->>RA: sources (cached 30 min, fallback chain)
    end
    RA->>RA: register sources → ids [1..n] · cited synthesis
    RA->>QH: post-research routing → blog
    QH->>BW: blog_writer
    BW->>SC: keyword_research(seed) (tool loop)
    BW->>BW: structured BlogDraft → meta/slug/H1 fixes · scrub URLs & ids · provenance
    BW->>QG: quality_target=blog
    alt score < threshold and revisions < 2 and under budget
        QG->>BW: suggestions (revision loop)
    else pass / max revisions
        QG->>FN: done
    end
    FN->>FN: output moderation · response · usage
    FN-->>U: blog + score + Sources & Provenance (AIMessage saved to memory)
```

## 4. Data and provenance flow

```mermaid
flowchart LR
    S1[SerpAPI] & S2[Tavily] & S3[Perplexity] --> SC[SearchChain] --> REG[(Source registry<br/>state.sources<br/>dedupe by URL · stable ids · used_in)]
    REG -->|numbered list, no URLs| LLM[Writers / Synthesis]
    LLM --> SCR[scrub_body<br/>drop invented URLs & ids]
    SCR --> ART[Artifact + Provenance<br/>sources · model · grounding · notes]
    REG --> ART
    ART --> UIX[Chat · Dashboard · Research panel · Exports]
```

## 5. Memory model

- `messages` (conversation) and every artifact are checkpointed per `thread_id`.
- `InMemorySaver` in development; `SqliteSaver` (`.cache/checkpoints.sqlite`) in production.
- Each Streamlit browser session gets its own thread id; "New conversation" starts a fresh thread.
- Tool-loop messages live in `agent_scratch`, which is cleared after each agent, so the memory holds only user
  turns and final assistant responses.
