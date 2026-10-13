"""Natural-language prompts used by the NAUTILUS CLIP detector.

Dataset labels remain unchanged for annotation/evaluation.  Only the strings
sent to CLIP are rewritten so abbreviations and taxonomy names become natural
language concepts that are better represented by CLIP pretraining.
"""


CLASS_NAME_ALIASES = {
    "corals": "coral",
    "fish": "fish",
    "cuttlefish": "cuttlefish",
    "diver": "scuba diver",
    "turtle": "sea turtle",
    "jellyfish": "jellyfish",
    "starfish": "starfish",
    "echinus": "sea urchin",
    "scallop": "scallop",
    "mollusk": "mollusk",
    "plants": "underwater plant",
    "reptiles": "marine reptile",
    "arthropoda": "marine arthropod",
    "human": "person",
    "garbage": "marine debris",
    "gbottle": "glass bottle",
    "pbag": "plastic bag",
    "stingray": "stingray",
    "ruins": "underwater ruins",
    "net": "fishing net",
    "shark": "shark",
    "puffin": "puffin",
    "robots": "underwater robot",
    "holothurian": "sea cucumber",
    "glove": "discarded glove",
    "tire": "discarded tire",
    "pbottle": "plastic bottle",
    "can": "discarded can",
    "mask": "face mask",
    "plastic": "plastic debris",
    "misc": "miscellaneous marine debris",
    "penguin": "penguin",
    "rod": "metal rod",
    "metal": "metal debris",
    "cellphone": "mobile phone",
    "electronics": "electronic waste",
    "small_fish": "small fish",
    "crab": "crab",
    "shrimp": "shrimp",
    "sunglasses": "sunglasses",
}


ARTICLE_FREE_CLASS_NAMES = {
    "coral",
    "marine debris",
    "underwater ruins",
    "plastic debris",
    "miscellaneous marine debris",
    "metal debris",
    "electronic waste",
    "small fish",
}


UNDERWATER_CLASS_TEMPLATES = (
    "an underwater photo of {}",
    "a photo of {} underwater",
    "{} in an underwater scene",
    "an underwater image containing {}",
)


UNDERWATER_BACKGROUND_TEMPLATES = (
    "an underwater background",
    "an underwater scene without a target object",
    "unknown underwater content",
)


def naturalize_class_name(class_name):
    """Return a natural-language CLIP name without changing the dataset ID."""
    raw_name = str(class_name).strip()
    lookup_key = raw_name.lower().replace(" ", "_")
    return CLASS_NAME_ALIASES.get(
        lookup_key,
        raw_name.replace("_", " ").lower(),
    )


def naturalize_class_phrase(class_name):
    """Return a grammatical noun phrase suitable for a CLIP sentence."""
    natural_name = naturalize_class_name(class_name)
    if natural_name in ARTICLE_FREE_CLASS_NAMES:
        return natural_name
    if natural_name == "sunglasses":
        return "a pair of sunglasses"
    article = "an" if natural_name[:1] in "aeiou" else "a"
    return "{} {}".format(article, natural_name)


def domain_class_description(class_name, domain_text):
    natural_name = naturalize_class_phrase(class_name)
    return "an underwater photo of {} with {}".format(
        natural_name, str(domain_text).strip()
    )


def domain_image_description(domain_text):
    return "an underwater image with {}".format(str(domain_text).strip())
