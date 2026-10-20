"""Agent 5 — Image Generator.

Pipeline:
1. **Prompt enhancement**: a fast LLM acts as art director. It turns the user's idea (or the
   campaign topic and blog title) into a detailed prompt covering subject, composition,
   lighting and mood, using the brand's palette and visual style. It also writes accessible
   alt text. If the LLM fails, a deterministic template is used instead.
2. **Safety**: the enhanced prompt goes through the moderation guardrail.
3. **Generation with fallback**: gpt-image-1 → gpt-image-1-mini → Gemini 2.5 Flash Image → branded Pillow placeholder.
4. **Validation & storage**: Pillow decodes and size-checks the image, which is saved as PNG to
   ``outputs/images/`` next to a JSON metadata file (prompts, provider, attempts, provenance).
"""

from __future__ import annotations

import json
import time

from pydantic import BaseModel

from src.agents import AgentDeps, add_output, add_warning, advance_pending, brand_of, request_of, topic_of
from src.core import prompts
from src.integrations.image_clients import ImageGenerationError
from src.utils.content_optimization import slugify
from src.utils.guardrails import check_output
from src.workflow.state_management import ContentState, ImageAsset, Provenance

# Appended to every prompt: image models often add lettering to signs and packaging unless told plainly not to.
NO_TEXT_SUFFIX = " Absolutely no text, letters, words, numbers, signs or logos anywhere in the image."


class ImagePromptSpec(BaseModel):
    enhanced_prompt: str
    alt_text: str = ""


def fallback_enhancement(idea: str, brand) -> ImagePromptSpec:
    prompt = (f"{idea}. {brand.visual_style}. Composition with clear focal point and negative space on one side "
              f"for a headline, soft natural lighting, shallow depth of field, color palette "
              f"{', '.join(brand.color_palette)}. High detail, professional marketing photograph. "
              "No text, no letters, no logos, no watermarks.")
    return ImagePromptSpec(enhanced_prompt=prompt[:900], alt_text=f"Marketing image: {idea}"[:125])


def make_image_generator(deps: AgentDeps):
    def image_generator(state: ContentState) -> dict:
        brand = brand_of(state)
        request = request_of(state)
        topic = topic_of(state)
        idea = request.image_prompt or topic
        context = []
        if blog := state.get("blog"):
            context.append(f"Hero image for the blog post '{blog.get('title', '')}'")
        if (linkedin := state.get("linkedin")) and not blog:
            context.append(f"Image to accompany a LinkedIn post that opens: '{linkedin.get('hook', '')}'")
        use_case = "a blog hero image and LinkedIn post (landscape)" if context else "social media and web (landscape)"
        warnings: list[str] = []

        spec = deps.llm.structured(
            ImagePromptSpec,
            prompts.IMAGE_PROMPT.format(visual_style=brand.visual_style, palette=", ".join(brand.color_palette),
                                        use_case=use_case),
            f"User idea: {idea}\nBrand: {brand.name} ({brand.industry})\n" + "\n".join(context),
            tier="fast",
        )
        if spec is None or not spec.enhanced_prompt.strip():
            spec = fallback_enhancement(idea, brand)
            warnings.append("Prompt enhancement model unavailable; used the built-in prompt template.")

        safe, categories = check_output(spec.enhanced_prompt, deps.moderator)
        if not safe:
            return {"warnings": add_warning(state, f"Image prompt blocked by moderation ({', '.join(categories)})."),
                    "pending_formats": advance_pending(state, "image")}

        try:
            result = deps.images.generate(spec.enhanced_prompt.rstrip() + NO_TEXT_SUFFIX, title=topic, palette=brand.color_palette,
                                          brand=brand.name)
        except ImageGenerationError as exc:
            return {"warnings": add_warning(state, f"Image generation failed: {exc}"),
                    "pending_formats": advance_pending(state, "image")}
        if result.attempts:
            warnings.append(f"Image fallback used → {result.provider} (failed: "
                            + "; ".join(a.split(':')[0] for a in result.attempts) + ").")

        folder = deps.output_dir / "images"
        folder.mkdir(parents=True, exist_ok=True)
        stem = f"{time.strftime('%Y%m%d-%H%M%S')}-{slugify(topic or 'image', 40) or 'image'}"
        path = folder / f"{stem}.png"
        path.write_bytes(result.png_bytes)

        provenance = Provenance(
            generated_by=deps.llm.describe("fast") + " (prompt enhancement)",
            image_model=result.provider,
            grounding="model_knowledge_only",
            notes=[f"Created for topic: {topic}", *[f"Fallback attempt failed — {a}" for a in result.attempts]],
        )
        asset = ImageAsset(path=str(path), original_prompt=idea, enhanced_prompt=spec.enhanced_prompt,
                           provider=result.provider, width=result.width, height=result.height,
                           alt_text=spec.alt_text or f"Image about {topic}"[:125], attempts=result.attempts,
                           provenance=provenance)
        meta_path = folder / f"{stem}.json"
        asset.metadata_path = str(meta_path)
        meta_path.write_text(json.dumps(asset.model_dump(), indent=2))

        return {
            "image": asset.model_dump(),
            "images": [*state.get("images", []), asset.model_dump()],
            "pending_formats": advance_pending(state, "image"),
            "outputs_this_turn": add_output(state, "image"),
            "warnings": add_warning(state, *warnings),
        }

    return image_generator
