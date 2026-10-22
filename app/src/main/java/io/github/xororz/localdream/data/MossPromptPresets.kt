package io.github.xororz.localdream.data

/** Curated scene-first examples, not SD1.5 boilerplate keyword spam.
 * MOSS's Qwen3.5 conditioner understands complete natural-language descriptions.
 * A preset only replaces the prompt when tapped; it never silently changes a draft.
 */
data class MossPromptPreset(val title: String, val category: String, val prompt: String)

object MossPromptPresets {
    val options = listOf(
        MossPromptPreset("Quiet afternoon", "Anime",
            "A hand-drawn 1990s Japanese slice-of-life anime film still. A thoughtful girl with short dark hair sits on the floor beside an open balcony door in a small lived-in apartment. A white cat sleeps next to a stack of paperback books. The last warm daylight falls across the tatami while distant apartment roofs sit softly out of focus. Clean expressive character drawing, carefully painted backgrounds, subtle cel shading, gentle atmospheric perspective, a believable ordinary moment."),
        MossPromptPreset("Train window", "Anime",
            "A beautifully composed hand-painted anime background of a nearly empty local train passing through the Japanese countryside on an overcast morning. A teenager in a navy cardigan rests their head against the window; rice fields, utility poles and distant misty hills slide past. Muted sage greens, soft diffuse daylight, detailed interiors, understated emotional storytelling, precise linework and delicate traditional animation color."),
        MossPromptPreset("Sunday market", "Photography",
            "A candid 35mm film photograph at a small outdoor farmers market in early autumn. An elderly florist in a worn linen apron wraps a bundle of white chrysanthemums with brown paper. The subject looks away from the camera, caught in a natural gesture. Cloud-filtered morning light, realistic skin and fabric textures, gentle film grain, natural colors, documentary composition, slightly imperfect framing, no studio retouching."),
        MossPromptPreset("After the rain", "Photography",
            "A natural street photograph in Kyoto just after rainfall. A small family-owned ramen shop with fogged windows, two bicycles by the entrance and a pedestrian carrying a folded umbrella. Soft gray daylight reflects from uneven stone pavement. Quiet everyday atmosphere, 50mm lens perspective, honest film colors, grounded architectural detail, believable scale and no dramatic artificial lighting."),
        MossPromptPreset("The archivist", "Fantasy",
            "An oil painting of a solitary archivist restoring a centuries-old map inside a vast medieval library. The shelves are asymmetrical and packed with weathered volumes, wax seals and brass instruments. A narrow beam of afternoon sunlight reveals dust above the desk. Rich but restrained earthy pigments, authentic wood and parchment textures, convincing architectural depth, painterly brushwork and quiet scholarly mystery."),
        MossPromptPreset("Mountain crossing", "Environment",
            "A wide landscape illustration of an old stone footbridge crossing a river in the Swiss Alps at the beginning of spring. Meltwater curls between smooth boulders; patches of snow remain on distant mountains and a tiny red-coated hiker gives the scene scale. Natural cool morning light, lush but realistic alpine vegetation, strong foreground-to-background depth, fine observational details and a calm, spacious composition."),
        MossPromptPreset("Editorial cover", "Design",
            "A sophisticated minimalist editorial book cover for an architecture journal. A single sculptural concrete staircase photographed from above against an off-white plaster wall. Warm sunlight creates one sharp geometric shadow. Strong negative space, disciplined modernist composition, tactile paper finish, charcoal-black elegant typography at the top reading 'FORM AND SPACE', and a tiny issue number in the lower corner. Precise readable layout."),
        MossPromptPreset("Ceramic still life", "Product",
            "A premium contemporary product photograph of a handmade ceramic teapot and two matching cups on a pale oak table. An imperfect linen cloth falls softly to one side, with one branch of olive leaves in the background. Diffused north-facing window light, subtle shadows, warm ivory and stone-gray palette, believable glazed ceramic surface, intentional negative space and restrained editorial styling.")
    )
}
