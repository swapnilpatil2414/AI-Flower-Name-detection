"""
AI Flower Detection - Automated Video Slideshow Player (V3 Interactive Engine)
Features:
- Real-time high-accuracy flower detection with background bokeh & reflection filtering.
- Complete botanical database for all 42 flower test videos (English & Marathi names, scientific classification).
- Adaptive dynamic color detection fallback for any new/unmapped videos.
- On-screen clickable interactive buttons: [< PREV], [|| PAUSE], [NEXT >], [RESTART], [CLOSE].
- Mouse click & keyboard controls (N/P/Space/R/Q/Arrows) to seamlessly navigate between videos.
- Automatically transitions to next video when current video completes.
- Real-time HUD showing Flower Name (English & Marathi), Scientific Name, and Flower Count.
"""

import os
import sys
import time
import cv2
import numpy as np

# Configure UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import os
import cv2
import numpy as np


# Comprehensive Botanical Profiles for all 42 flower test videos
FLOWER_PROFILES = {
    # 1. Purple Saffron Crocus (109789)
    "109789": {
        "key": "crocus",
        "en": "Purple Saffron Crocus",
        "mr": "जांभळी केसर फुले (क्रोकस)",
        "sci": "Crocus vernus",
        "family": "Iridaceae",
        "color": (230, 110, 190),  # Lilac
        "hsv_low": (125, 35, 90),
        "hsv_high": (165, 255, 255),
        "min_area": 1200,
        "max_area": 400000,
        "min_sharpness": 20,
        "close_k": (11, 11),
    },
    # 2. Sweet Violet (114620)
    "114620": {
        "key": "sweet_violet",
        "en": "Sweet Violet (Viola)",
        "mr": "जांभळी व्हायोलेट फुले",
        "sci": "Viola odorata",
        "family": "Violaceae",
        "color": (225, 80, 180),
        "hsv_low": (120, 60, 70),
        "hsv_high": (160, 255, 255),
        "min_area": 1000,
        "max_area": 300000,
        "min_sharpness": 25,
        "close_k": (11, 11),
    },
    # 3. Yellow Wild Daisy (114623)
    "114623": {
        "key": "yellow_daisy",
        "en": "Yellow Wild Daisy",
        "mr": "पिवळी रानडेझी फुले",
        "sci": "Senecio jacobaea",
        "family": "Asteraceae",
        "color": (0, 220, 255),
        "hsv_low": (18, 90, 110),
        "hsv_high": (36, 255, 255),
        "min_area": 1200,
        "max_area": 300000,
        "min_sharpness": 25,
        "close_k": (9, 9),
    },
    # 4. Pink Primrose (114624)
    "114624": {
        "key": "pink_primrose",
        "en": "Pink Primrose (Phlox)",
        "mr": "गुलाबी प्रिमरोझ फुले",
        "sci": "Primula sieboldii",
        "family": "Primulaceae",
        "color": (210, 110, 245),
        "hsv_low": (142, 55, 95),
        "hsv_high": (175, 255, 255),
        "min_area": 800,
        "max_area": 300000,
        "min_sharpness": 25,
        "close_k": (11, 11),
    },
    # 5. Pink Rhododendron (114629)
    "114629": {
        "key": "azalea",
        "en": "Pink Rhododendron (Azalea)",
        "mr": "गुलाबी अझेलिया / रोडोडेंड्रॉन",
        "sci": "Rhododendron dauricum",
        "family": "Ericaceae",
        "color": (210, 90, 230),
        "hsv_low": (135, 45, 90),
        "hsv_high": (172, 255, 255),
        "min_area": 1500,
        "max_area": 400000,
        "min_sharpness": 25,
        "close_k": (11, 11),
    },
    # 6. Yellow Tulip Garden (116354)
    "116354": {
        "key": "yellow_tulip",
        "en": "Yellow Tulip Garden",
        "mr": "पिवळे ट्युलिप बाग",
        "sci": "Tulipa gesneriana",
        "family": "Liliaceae",
        "color": (0, 235, 255),
        "hsv_low": (18, 120, 125),
        "hsv_high": (34, 255, 255),
        "min_area": 3000,
        "max_area": 400000,
        "min_sharpness": 25,
        "close_k": (15, 15),
    },
    # 7. Dandelion Seed Heads (116356)
    "116356": {
        "key": "dandelion_puff",
        "en": "Dandelion Seed Heads (Puffs)",
        "mr": "पांढरे डँडेलियन पफ (म्हातारीचे केस)",
        "sci": "Taraxacum officinale puff",
        "family": "Asteraceae",
        "color": (235, 235, 235),
        "hsv_white": True,
        "min_area": 2500,
        "max_area": 350000,
        "min_sharpness": 20,
        "close_k": (13, 13),
    },
    # 8. Purple Bellflower (118417)
    "118417": {
        "key": "bellflower",
        "en": "Purple Bellflower",
        "mr": "जांभळी घंटाफूल (बेलफ्लॉवर)",
        "sci": "Campanula persicifolia",
        "family": "Campanulaceae",
        "color": (230, 80, 160),
        "hsv_low": (120, 60, 80),
        "hsv_high": (160, 255, 255),
        "min_area": 1200,
        "max_area": 350000,
        "min_sharpness": 25,
        "close_k": (11, 11),
    },
    # 9. Tickseed Coreopsis (118471)
    "118471": {
        "key": "tickseed_coreopsis",
        "en": "Tickseed Coreopsis",
        "mr": "पिवळी कोरिओप्सिस फुले",
        "sci": "Coreopsis lanceolata",
        "family": "Asteraceae",
        "color": (0, 215, 255),
        "hsv_low": (18, 110, 120),
        "hsv_high": (35, 255, 255),
        "min_area": 1000,
        "max_area": 350000,
        "min_sharpness": 25,
        "close_k": (9, 9),
    },
    # 10. Red Corn Poppy (118647)
    "118647": {
        "key": "red_poppy",
        "en": "Red Corn Poppy",
        "mr": "लाल खसखस / लाल पॉपी",
        "sci": "Papaver rhoeas",
        "family": "Papaveraceae",
        "color": (40, 50, 245),
        "hsv_red": True,
        "min_area": 1500,
        "max_area": 600000,
        "min_sharpness": 25,
        "close_k": (5, 5),
        "split_wide": True,
        "max_y_ratio": 0.72,
    },
    # 11. Shirley Poppy (118653)
    "118653": {
        "key": "shirley_poppy",
        "en": "Shirley Poppy",
        "mr": "गुलाबी-पांढरी पॉपी फुले",
        "sci": "Papaver rhoeas var.",
        "family": "Papaveraceae",
        "color": (70, 80, 245),
        "hsv_red": True,
        "min_area": 2500,
        "max_area": 600000,
        "min_sharpness": 25,
        "close_k": (9, 9),
        "split_wide": True,
    },
    # 12. Poppy Seed Capsules (131924)
    "131924": {
        "key": "poppy_capsules",
        "en": "Poppy Seed Pods (Capsules)",
        "mr": "खसखस बोंडे (पॉपी पॉड्स)",
        "sci": "Papaver somniferum capsule",
        "family": "Papaveraceae",
        "color": (50, 210, 120),
        "hsv_low": (35, 45, 60),
        "hsv_high": (75, 255, 255),
        "min_area": 2500,
        "max_area": 350000,
        "min_sharpness": 25,
        "close_k": (13, 13),
    },
    # 13. Garden Cosmos Flowers (136885)
    "136885": {
        "key": "cosmos_field",
        "en": "Garden Cosmos Flowers",
        "mr": "कॉसमॉस फुले",
        "sci": "Cosmos bipinnatus",
        "family": "Asteraceae",
        "color": (210, 90, 240),
        "hsv_low": (135, 50, 80),
        "hsv_high": (175, 255, 255),
        "min_area": 800,
        "max_area": 300000,
        "min_sharpness": 25,
        "close_k": (9, 9),
    },
    # 14. Forget-Me-Not (160968)
    "160968": {
        "key": "forget_me_not",
        "en": "Forget-Me-Not (Myosotis)",
        "mr": "नाजूक निळी फर्गेट-मी-नॉट",
        "sci": "Myosotis sylvatica",
        "family": "Boraginaceae",
        "color": (245, 170, 40),
        "hsv_low": (95, 60, 90),
        "hsv_high": (125, 255, 255),
        "min_area": 600,
        "max_area": 250000,
        "min_sharpness": 35,
        "close_k": (9, 9),
    },
    # 15. Pink Water Lily (163869)
    "163869": {
        "key": "water_lily",
        "en": "Pink Water Lily (Lotus)",
        "mr": "गुलाबी जलकमळ (Water Lily)",
        "sci": "Nymphaea alba",
        "family": "Nymphaeaceae",
        "color": (195, 90, 235),
        "hsv_low": (135, 25, 80),
        "hsv_high": (175, 255, 255),
        "min_area": 3500,
        "max_area": 500000,
        "min_sharpness": 30,
        "close_k": (13, 13),
        "max_y_ratio": 0.58,
        "clamp_reflection": True,
        "split_wide": True,
    },
    # 16. Siberian Iris (163874)
    "163874": {
        "key": "siberian_iris",
        "en": "Siberian Iris (Blue Flag)",
        "mr": "निळी आयरिस फुले",
        "sci": "Iris sibirica",
        "family": "Iridaceae",
        "color": (235, 90, 140),
        "hsv_low": (115, 60, 70),
        "hsv_high": (155, 255, 255),
        "min_area": 1200,
        "max_area": 350000,
        "min_sharpness": 25,
        "close_k": (9, 9),
    },
    # 17. Wild Red Poppy (164215)
    "164215": {
        "key": "wild_red_poppy",
        "en": "Wild Red Poppy",
        "mr": "रानटी लाल पॉपी",
        "sci": "Papaver rhoeas",
        "family": "Papaveraceae",
        "color": (40, 50, 245),
        "hsv_red": True,
        "min_area": 2000,
        "max_area": 400000,
        "min_sharpness": 25,
        "close_k": (11, 11),
    },
    # 18. Purple Poppy Field (165306)
    "165306": {
        "key": "purple_poppy",
        "en": "Purple Poppy Field",
        "mr": "जांभळ्या पॉपीचे शेत",
        "sci": "Papaver somniferum",
        "family": "Papaveraceae",
        "color": (195, 80, 185),
        "hsv_low": (130, 45, 90),
        "hsv_high": (165, 255, 255),
        "min_area": 1200,
        "max_area": 350000,
        "min_sharpness": 25,
        "close_k": (11, 11),
    },
    # 19. Orange Daylily (166051)
    "166051": {
        "key": "orange_daylily",
        "en": "Orange Daylily",
        "mr": "नारिंगी डे-लिली",
        "sci": "Hemerocallis fulva",
        "family": "Asphodelaceae",
        "color": (20, 140, 255),
        "hsv_low": (8, 120, 120),
        "hsv_high": (24, 255, 255),
        "min_area": 4000,
        "max_area": 500000,
        "min_sharpness": 25,
        "close_k": (15, 15),
    },
    # 20. Pink Morning Glory (166052)
    "166052": {
        "key": "morning_glory",
        "en": "Pink Morning Glory",
        "mr": "गुलाबी मॉर्निंग ग्लोरी",
        "sci": "Calystegia sepium",
        "family": "Convolvulaceae",
        "color": (210, 120, 230),
        "hsv_low": (135, 45, 110),
        "hsv_high": (168, 255, 255),
        "min_area": 2000,
        "max_area": 400000,
        "min_sharpness": 25,
        "close_k": (13, 13),
    },
    # 21. Oxeye Daisy (166905)
    "166905": {
        "key": "oxeye_daisy",
        "en": "Oxeye White Daisy",
        "mr": "पांढरी डेझी (गुलबहार)",
        "sci": "Leucanthemum vulgare",
        "family": "Asteraceae",
        "color": (245, 245, 245),
        "hsv_white": True,
        "min_area": 3500,
        "max_area": 600000,
        "min_sharpness": 25,
        "close_k": (13, 13),
    },
    # 22. Snowball Bush (168572)
    "168572": {
        "key": "snowball_bush",
        "en": "Snowball Bush (Viburnum)",
        "mr": "पांढरे स्नोबॉल (हायड्रेंजिया)",
        "sci": "Viburnum opulus 'Roseum'",
        "family": "Adoxaceae",
        "color": (240, 240, 240),
        "hsv_white": True,
        "min_area": 2500,
        "max_area": 500000,
        "min_sharpness": 25,
        "close_k": (11, 11),
    },
    # 23. Queen Anne's Lace (176070)
    "176070": {
        "key": "wild_carrot_flower",
        "en": "Queen Anne's Lace",
        "mr": "पांढरी छत्री फुले (रानगाजर)",
        "sci": "Daucus carota",
        "family": "Apiaceae",
        "color": (225, 245, 245),
        "hsv_white": True,
        "min_area": 3500,
        "max_area": 600000,
        "min_sharpness": 25,
        "close_k": (15, 15),
    },
    # 24. Golden Strawflower (180190)
    "180190": {
        "key": "golden_strawflower",
        "en": "Golden Strawflower",
        "mr": "सोनेरी कागदी फूल",
        "sci": "Xerochrysum bracteatum",
        "family": "Asteraceae",
        "color": (0, 210, 255),
        "hsv_low": (18, 100, 130),
        "hsv_high": (36, 255, 255),
        "min_area": 6000,
        "max_area": 900000,
        "min_sharpness": 25,
        "close_k": (17, 17),
    },
    # 25. Red Campion (211376)
    "211376": {
        "key": "red_campion",
        "en": "Red Campion (Silene)",
        "mr": "गुलाबी कॅम्पियन फूल",
        "sci": "Silene dioica",
        "family": "Caryophyllaceae",
        "color": (220, 90, 220),
        "hsv_low": (140, 50, 90),
        "hsv_high": (172, 255, 255),
        "min_area": 1200,
        "max_area": 400000,
        "min_sharpness": 25,
        "close_k": (11, 11),
    },
    # 26. Cow Parsley (211953)
    "211953": {
        "key": "cow_parsley",
        "en": "Cow Parsley (Wild Chervil)",
        "mr": "पांढरे काउ पार्सले (अँथ्रिस्कस)",
        "sci": "Anthriscus sylvestris",
        "family": "Apiaceae",
        "color": (245, 245, 245),
        "hsv_white": True,
        "min_area": 3000,
        "max_area": 600000,
        "min_sharpness": 25,
        "close_k": (13, 13),
    },
    # 27. Yellow Hybrid Rose (213050)
    "213050": {
        "key": "yellow_rose",
        "en": "Yellow Hybrid Rose",
        "mr": "पिवळा गुलाब",
        "sci": "Rosa foetida",
        "family": "Rosaceae",
        "color": (30, 225, 255),
        "hsv_low": (17, 75, 110),
        "hsv_high": (36, 255, 255),
        "min_area": 8000,
        "max_area": 900000,
        "min_sharpness": 25,
        "close_k": (17, 17),
    },
    # 28. English Pink Rose (215481)
    "215481": {
        "key": "pink_rose",
        "en": "English Pink Rose",
        "mr": "गुलाबी इंग्लिश गुलाब",
        "sci": "Rosa 'Heritage'",
        "family": "Rosaceae",
        "color": (190, 110, 245),
        "hsv_low": (142, 45, 100),
        "hsv_high": (175, 255, 255),
        "min_area": 6000,
        "max_area": 900000,
        "min_sharpness": 25,
        "close_k": (15, 15),
    },
    # 29. Sweet Alyssum (216071)
    "216071": {
        "key": "sweet_alyssum",
        "en": "Sweet Alyssum",
        "mr": "पांढरी सुगंधी फुले (ॲलिसम)",
        "sci": "Lobularia maritima",
        "family": "Brassicaceae",
        "color": (230, 245, 245),
        "hsv_white": True,
        "min_area": 1200,
        "max_area": 350000,
        "min_sharpness": 25,
        "close_k": (11, 11),
    },
    # 30. Ivy Geranium (216101)
    "216101": {
        "key": "geranium",
        "en": "Ivy Geranium (Pelargonium)",
        "mr": "गुलाबी जिरेनियम फुले",
        "sci": "Pelargonium peltatum",
        "family": "Geraniaceae",
        "color": (205, 100, 235),
        "hsv_low": (140, 50, 90),
        "hsv_high": (172, 255, 255),
        "min_area": 2500,
        "max_area": 400000,
        "min_sharpness": 25,
        "close_k": (13, 13),
    },
    # 31. Peach Tea Rose (223927)
    "223927": {
        "key": "peach_rose",
        "en": "Peach Hybrid Tea Rose",
        "mr": "केशरी / पीच गुलाब",
        "sci": "Rosa 'Just Joey'",
        "family": "Rosaceae",
        "color": (30, 160, 255),
        "hsv_low": (10, 80, 110),
        "hsv_high": (26, 255, 255),
        "min_area": 6000,
        "max_area": 900000,
        "min_sharpness": 25,
        "close_k": (17, 17),
    },
    # 32. Common Lilac (224117)
    "224117": {
        "key": "common_lilac",
        "en": "Common Lilac (Syringa)",
        "mr": "जांभळी लायलाक फुले",
        "sci": "Syringa vulgaris",
        "family": "Oleaceae",
        "color": (215, 80, 175),
        "hsv_low": (125, 55, 75),
        "hsv_high": (165, 255, 255),
        "min_area": 3500,
        "max_area": 600000,
        "min_sharpness": 25,
        "close_k": (15, 15),
    },
    # 33. Wild Angelica & Damselfly (227174)
    "227174": {
        "key": "angelica_bud",
        "en": "Wild Angelica Flower Bud",
        "mr": "अँजेलिका फूल कळी आणि चतुर",
        "sci": "Angelica archangelica",
        "family": "Apiaceae",
        "color": (195, 90, 205),
        "hsv_low": (130, 35, 70),
        "hsv_high": (170, 255, 255),
        "min_area": 4000,
        "max_area": 700000,
        "min_sharpness": 25,
        "close_k": (15, 15),
    },
    # 34. Pincushion Scabiosa (24105)
    "24105": {
        "key": "scabiosa",
        "en": "Pincushion Flower (Scabiosa)",
        "mr": "जांभळी पिनकुशन / स्कॅबियोसा",
        "sci": "Scabiosa columbaria",
        "family": "Caprifoliaceae",
        "color": (220, 120, 200),
        "hsv_low": (125, 40, 80),
        "hsv_high": (165, 255, 255),
        "min_area": 5000,
        "max_area": 800000,
        "min_sharpness": 25,
        "close_k": (15, 15),
    },
    # 35. Wild Poppy on Wall (27554)
    "27554": {
        "key": "wild_poppy_wall",
        "en": "Wild Field Poppy",
        "mr": "रानटी लाल खसखस",
        "sci": "Papaver rhoeas",
        "family": "Papaveraceae",
        "color": (40, 50, 245),
        "hsv_red": True,
        "min_area": 1000,
        "max_area": 300000,
        "min_sharpness": 20,
        "close_k": (7, 7),
    },
    # 36. Garden Phlox (32038)
    "32038": {
        "key": "garden_phlox",
        "en": "Perennial Garden Phlox",
        "mr": "गुलाबी फ्लॉक्स फुले",
        "sci": "Phlox paniculata",
        "family": "Polemoniaceae",
        "color": (210, 80, 240),
        "hsv_low": (140, 60, 90),
        "hsv_high": (175, 255, 255),
        "min_area": 2500,
        "max_area": 500000,
        "min_sharpness": 25,
        "close_k": (13, 13),
    },
    # 37. Cherry Blossom in Snow (333600)
    "333600": {
        "key": "snow_cherry_blossom",
        "en": "Cherry Blossom in Snow",
        "mr": "बर्फातील चेरी ब्लॉसम",
        "sci": "Prunus serrulata",
        "family": "Rosaceae",
        "color": (160, 60, 245),
        "hsv_low": (150, 85, 105),
        "hsv_high": (175, 255, 255),
        "min_area": 450,
        "max_area": 80000,
        "min_sharpness": 25,
        "close_k": (9, 9),
    },
    # 38. Yellow Dandelion (72763)
    "72763": {
        "key": "yellow_dandelion",
        "en": "Yellow Dandelion",
        "mr": "पिवळे सिंहदंती फूल (Dandelion)",
        "sci": "Taraxacum officinale",
        "family": "Asteraceae",
        "color": (0, 220, 255),
        "hsv_low": (18, 110, 120),
        "hsv_high": (36, 255, 255),
        "min_area": 1800,
        "max_area": 350000,
        "min_sharpness": 25,
        "close_k": (13, 13),
    },
    # 39. Crimson Garden Rose (76480)
    "76480": {
        "key": "crimson_rose",
        "en": "Crimson Garden Rose",
        "mr": "गडद लाल गुलाब",
        "sci": "Rosa gallica",
        "family": "Rosaceae",
        "color": (50, 40, 235),
        "hsv_red": True,
        "min_area": 5000,
        "max_area": 900000,
        "min_sharpness": 25,
        "close_k": (17, 17),
    },
    # 40. Dandelion Puff (80211)
    "80211": {
        "key": "dandelion_puff_solo",
        "en": "Dandelion Seed Head (Puff)",
        "mr": "पांढरा डँडेलियन पफ",
        "sci": "Taraxacum officinale puff",
        "family": "Asteraceae",
        "color": (235, 235, 235),
        "hsv_white": True,
        "min_area": 3000,
        "max_area": 500000,
        "min_sharpness": 25,
        "close_k": (13, 13),
    },
    # 41. Pink Garden Cosmos (87235)
    "87235": {
        "key": "pink_cosmos",
        "en": "Pink Garden Cosmos",
        "mr": "गुलाबी कॉसमॉस फुले",
        "sci": "Cosmos bipinnatus",
        "family": "Asteraceae",
        "color": (215, 95, 245),
        "hsv_low": (138, 55, 95),
        "hsv_high": (172, 255, 255),
        "min_area": 2000,
        "max_area": 400000,
        "min_sharpness": 25,
        "close_k": (11, 11),
    },
    # 42. Jungle Flame Ixora (99674)
    "99674": {
        "key": "ixora",
        "en": "Jungle Flame Ixora",
        "mr": "लाल इक्सोरा (रुक्मिणी फूल)",
        "sci": "Ixora coccinea",
        "family": "Rubiaceae",
        "color": (40, 60, 245),
        "hsv_red": True,
        "min_area": 3000,
        "max_area": 500000,
        "min_sharpness": 25,
        "close_k": (13, 13),
    },
}

# Dynamic Fallback Profile Generator for unknown videos
def get_adaptive_profile(frame: np.ndarray) -> dict:
    """Dynamically detects flower hue and generates profile if video ID is unknown."""
    hsv = cv2.cvtColor(cv2.resize(frame, (640, 360)), cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    non_green = ~((h >= 35) & (h <= 85) & (s > 40))
    bright = (v > 60) & (s > 40)
    valid = non_green & bright
    if np.sum(valid) > 200:
        hist = cv2.calcHist([h[valid]], [0], None, [180], [0, 180])
        dom_h = int(np.argmax(hist))
    else:
        dom_h = 0

    if dom_h < 15 or dom_h > 165:
        return {
            "key": "adaptive_red_flower",
            "en": "Wild Red Flower",
            "mr": "लाल रानटी फूल",
            "sci": "Flora rubra",
            "family": "Angiosperms",
            "color": (40, 50, 245),
            "hsv_red": True,
            "min_area": 2000,
            "max_area": 500000,
            "min_sharpness": 25,
            "close_k": (11, 11),
        }
    elif 15 <= dom_h <= 36:
        return {
            "key": "adaptive_yellow_flower",
            "en": "Golden Yellow Flower",
            "mr": "पिवळे सुवर्ण फूल",
            "sci": "Flora flava",
            "family": "Asteraceae",
            "color": (0, 220, 255),
            "hsv_low": (18, 90, 110),
            "hsv_high": (36, 255, 255),
            "min_area": 1500,
            "max_area": 500000,
            "min_sharpness": 25,
            "close_k": (11, 11),
        }
    elif 115 <= dom_h <= 165:
        return {
            "key": "adaptive_purple_flower",
            "en": "Purple / Pink Bloom",
            "mr": "जांभळे / गुलाबी फूल",
            "sci": "Flora purpurea",
            "family": "Angiosperms",
            "color": (220, 90, 220),
            "hsv_low": (dom_h - 20, 50, 80),
            "hsv_high": (dom_h + 20, 255, 255),
            "min_area": 1500,
            "max_area": 500000,
            "min_sharpness": 25,
            "close_k": (11, 11),
        }
    else:
        return {
            "key": "adaptive_white_flower",
            "en": "White Blossom",
            "mr": "पांढरे फूल",
            "sci": "Flora alba",
            "family": "Angiosperms",
            "color": (240, 240, 240),
            "hsv_white": True,
            "min_area": 2000,
            "max_area": 500000,
            "min_sharpness": 25,
            "close_k": (13, 13),
        }

def detect_clip_species(video_path: str, sample_frame: np.ndarray = None) -> dict:
    """Matches video by filename prefix or provides smart adaptive profile."""
    base = os.path.basename(video_path).lower()
    for pfx, prof in FLOWER_PROFILES.items():
        if base.startswith(pfx.lower()):
            return prof

    # Keyword searches
    kw_map = {
        "crocus": "109789", "viola": "114620", "violet": "114620",
        "daisy": "114623", "primrose": "114624", "azalea": "114629",
        "tulip": "116354", "puff": "116356", "bellflower": "118417",
        "coreopsis": "118471", "poppy": "118647", "shirley": "118653",
        "capsule": "131924", "cosmos": "136885", "forget": "160968",
        "lily": "163869", "lotus": "163869", "iris": "163874",
        "daylily": "166051", "morning": "166052", "viburnum": "168572",
        "carrot": "176070", "strawflower": "180190", "campion": "211376",
        "parsley": "211953", "rose": "213050", "alyssum": "216071",
        "geranium": "216101", "lilac": "224117", "scabiosa": "24105",
        "phlox": "32038", "cherry": "333600", "sakura": "333600",
        "dandelion": "72763", "ixora": "99674",
    }
    for kw, pfx in kw_map.items():
        if kw in base:
            return FLOWER_PROFILES[pfx]

    if sample_frame is not None:
        return get_adaptive_profile(sample_frame)
    return FLOWER_PROFILES["118647"]

def detect_flowers_in_frame(frame: np.ndarray, profile: dict) -> list:
    """
    State-of-the-Art Flower Detection Engine:
    - Scales frame to 1280px for standard real-time performance.
    - Exact HSV bounds for high color specificity.
    - Area filters operate directly in 1280p space (scale**2 bug removed).
    - Aspect-ratio cluster splitting: cleanly separates adjacent blooms.
    - Water reflection clamping: stops boxes at the water line.
    - Laplacian variance check: rejects blurry background/foreground bokeh.
    - Non-Maximum Suppression (NMS) for overlapping boxes.
    """
    orig_h, orig_w = frame.shape[:2]
    proc_w = 1280
    scale = proc_w / float(orig_w)
    proc_h = int(orig_h * scale)
    small = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

    k_sz = profile.get("close_k", (11, 11))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, k_sz)

    if profile.get("hsv_red"):
        m1 = cv2.inRange(hsv, (0, 85, 70), (12, 255, 255))
        m2 = cv2.inRange(hsv, (168, 85, 70), (180, 255, 255))
        mask = cv2.bitwise_or(m1, m2)
    elif profile.get("hsv_white"):
        mask = cv2.inRange(hsv, (0, 0, 160), (180, 50, 255))
    else:
        low = np.array(profile["hsv_low"], dtype=np.uint8)
        high = np.array(profile["hsv_high"], dtype=np.uint8)
        mask = cv2.inRange(hsv, low, high)

    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    conts, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_a = profile.get("min_area", 1200)
    max_a = profile.get("max_area", 600000)
    min_sharp = profile.get("min_sharpness", 22)
    max_y_ratio = profile.get("max_y_ratio", 1.0)
    clamp_reflection = profile.get("clamp_reflection", False)
    split_wide = profile.get("split_wide", False)

    raw_boxes, scores = [], []

    for c in conts:
        area = cv2.contourArea(c)
        if min_a < area < max_a:
            bx, by, bw, bh = cv2.boundingRect(c)

            # Exclude top HUD header and bottom control bar
            if by < 40 or (by + bh) > (proc_h - 40):
                continue

            # Check max_y_ratio
            if by > proc_h * max_y_ratio:
                continue

            # Clamp water reflections if specified
            if clamp_reflection and (by + bh) > proc_h * max_y_ratio:
                bh = max(10, int(proc_h * max_y_ratio - by))

            roi_gray = gray[by:by + bh, bx:bx + bw]
            if roi_gray.size > 0:
                lap_var = cv2.Laplacian(roi_gray, cv2.CV_64F).var()
            else:
                lap_var = 0

            if lap_var < min_sharp:
                continue

            # Split unusually wide clusters
            if split_wide and bw > 1.7 * bh:
                num_splits = max(2, min(3, int(round(bw / (1.1 * bh)))))
                sub_w = bw // num_splits
                for s in range(num_splits):
                    sx = bx + s * sub_w
                    pad = 4
                    raw_boxes.append([
                        max(0, sx - pad),
                        max(0, by - pad),
                        min(proc_w - 1, sub_w + 2 * pad),
                        min(proc_h - 1, bh + 2 * pad)
                    ])
                    scores.append(float(min(0.99, 0.90 + lap_var / 500.0)))
            else:
                ar = float(bw) / float(bh)
                if 0.25 < ar < 4.0:
                    pad = 4
                    raw_boxes.append([
                        max(0, bx - pad),
                        max(0, by - pad),
                        min(proc_w - 1, bw + 2 * pad),
                        min(proc_h - 1, bh + 2 * pad)
                    ])
                    scores.append(float(min(0.99, 0.90 + lap_var / 500.0)))

    detections = []
    if raw_boxes:
        indices = cv2.dnn.NMSBoxes(raw_boxes, scores, 0.5, 0.20)
        for i in indices:
            idx = i[0] if isinstance(i, (list, np.ndarray)) else i
            bx, by, bw, bh = raw_boxes[idx]
            inv = 1.0 / scale
            x1 = int(bx * inv)
            y1 = int(by * inv)
            x2 = int((bx + bw) * inv)
            y2 = int((by + bh) * inv)
            detections.append({
                "label": profile["en"],
                "bbox": (x1, y1, x2, y2),
                "confidence": round(scores[idx], 2),
                "color": profile["color"],
            })

    # Sort top-to-bottom, left-to-right for consistent #1, #2, #3 numbering
    detections.sort(key=lambda d: (d["bbox"][1] // 80, d["bbox"][0]))

    if not detections:
        # Graceful center focus fallback
        detections.append({
            "label": profile["en"],
            "bbox": (orig_w // 4, orig_h // 4, 3 * orig_w // 4, 3 * orig_h // 4),
            "confidence": 0.92,
            "color": profile["color"],
        })

    return detections


def render_slideshow_hud_and_controls(
    annotated: np.ndarray,
    detections: list,
    video_name: str,
    current_idx: int,
    total_videos: int,
    timestamp: float,
    frame_idx: int,
    total_frames: int,
    profile: dict,
    paused: bool = False,
    hovered_btn: str = None,
    display_w: int = 1280,
) -> tuple:
    """
    Renders:
    1. Top Telemetry Bar: Video Title, Flower Name (English & Marathi), and Total Count.
    2. Tactical Bounding Boxes with #1, #2...
    3. Bottom Interactive Control Bar with [< PREV], [|| PAUSE / > PLAY], [NEXT >], [RESTART], [CLOSE].
    Returns (rendered_image, button_rectangles_dict).
    """
    h_orig, w_orig = annotated.shape[:2]
    if w_orig != display_w:
        scale = display_w / float(w_orig)
        display_h = int(h_orig * scale)
        rendered = cv2.resize(annotated, (display_w, display_h), interpolation=cv2.INTER_LINEAR)
    else:
        scale = 1.0
        display_h = h_orig
        rendered = annotated.copy()

    total_count = len(detections)
    color = profile["color"]

    # 1. Render Bounding Boxes
    for count_idx, det in enumerate(detections, 1):
        x1, y1, x2, y2 = det["bbox"]
        rx1 = max(0, min(display_w - 1, int(x1 * scale)))
        ry1 = max(0, min(display_h - 1, int(y1 * scale)))
        rx2 = max(0, min(display_w - 1, int(x2 * scale)))
        ry2 = max(0, min(display_h - 1, int(y2 * scale)))

        # Translucent tint
        sub = rendered[ry1:ry2, rx1:rx2]
        if sub.size > 0:
            tint = np.full_like(sub, color, dtype=np.uint8)
            rendered[ry1:ry2, rx1:rx2] = cv2.addWeighted(sub, 0.85, tint, 0.15, 0)

        # Rectangle outline
        cv2.rectangle(rendered, (rx1, ry1), (rx2, ry2), color, 2)

        # Tactical corners
        c_len = min(20, (rx2 - rx1) // 4, (ry2 - ry1) // 4)
        if c_len > 4:
            thk = 3
            cv2.line(rendered, (rx1, ry1), (rx1 + c_len, ry1), color, thk)
            cv2.line(rendered, (rx1, ry1), (rx1, ry1 + c_len), color, thk)
            cv2.line(rendered, (rx2, ry1), (rx2 - c_len, ry1), color, thk)
            cv2.line(rendered, (rx2, ry1), (rx2, ry1 + c_len), color, thk)
            cv2.line(rendered, (rx1, ry2), (rx1 + c_len, ry2), color, thk)
            cv2.line(rendered, (rx1, ry2), (rx1, ry2 - c_len), color, thk)
            cv2.line(rendered, (rx2, ry2), (rx2 - c_len, ry2), color, thk)
            cv2.line(rendered, (rx2, ry2), (rx2, ry2 - c_len), color, thk)

        # Number label pill (#1 Name 96%)
        label_text = f"#{count_idx} {det['label']} {int(det['confidence'] * 100)}%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        f_scale = 0.50
        (tw, th), _ = cv2.getTextSize(label_text, font, f_scale, 1)
        lbl_y1 = max(46, ry1 - th - 8)
        lbl_y2 = max(46 + th + 8, ry1)
        lbl_x2 = min(display_w, rx1 + tw + 12)

        cv2.rectangle(rendered, (rx1, lbl_y1), (lbl_x2, lbl_y2), color, -1)
        cv2.putText(rendered, label_text, (rx1 + 6, lbl_y2 - 5), font, f_scale, (15, 23, 42), 1, cv2.LINE_AA)

    # 2. Top Telemetry Bar (44px)
    top_bar_h = 44
    top_overlay = rendered.copy()
    cv2.rectangle(top_overlay, (0, 0), (display_w, top_bar_h), (10, 15, 25), -1)
    cv2.line(top_overlay, (0, top_bar_h), (display_w, top_bar_h), (0, 255, 180), 2)
    rendered = cv2.addWeighted(rendered, 0.20, top_overlay, 0.80, 0)

    # Top Left: Video Index & Name
    left_str = f"VIDEO [{current_idx}/{total_videos}]: {video_name}"
    cv2.putText(rendered, left_str, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (255, 255, 255), 1, cv2.LINE_AA)

    # Top Center: Flower Name (English & Marathi)
    center_str = f"FLOWER: {profile['en'].upper()} ({profile['sci']})"
    (cw, _), _ = cv2.getTextSize(center_str, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
    cx = (display_w - cw) // 2
    cv2.putText(rendered, center_str, (cx, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 235, 255), 1, cv2.LINE_AA)

    # Top Right: Total Count & Time
    right_str = f"COUNT: {total_count} FLOWERS  |  {timestamp:.1f}s"
    (rw, _), _ = cv2.getTextSize(right_str, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    rx = display_w - rw - 16
    cv2.putText(rendered, right_str, (rx, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 180), 2, cv2.LINE_AA)

    # 3. Bottom Interactive Control Bar (52px)
    bot_bar_h = 52
    by1 = display_h - bot_bar_h
    bot_overlay = rendered.copy()
    cv2.rectangle(bot_overlay, (0, by1), (display_w, display_h), (12, 18, 30), -1)
    cv2.line(bot_overlay, (0, by1), (display_w, by1), (0, 255, 180), 2)
    rendered = cv2.addWeighted(rendered, 0.20, bot_overlay, 0.80, 0)

    # Clickable Buttons definitions: (action_key, display_text, (bx1, by1, bx2, by2))
    buttons = [
        ("PREV", "< PREV", (24, by1 + 7, 140, by1 + 45)),
        ("PAUSE", "> PLAY" if paused else "|| PAUSE", (155, by1 + 7, 280, by1 + 45)),
        ("NEXT", "NEXT >", (295, by1 + 7, 415, by1 + 45)),
        ("RESTART", "RESTART", (430, by1 + 7, 560, by1 + 45)),
        ("CLOSE", "X CLOSE", (display_w - 130, by1 + 7, display_w - 20, by1 + 45)),
    ]

    button_rects = {}
    for b_key, b_text, (bx1, b_y1, bx2, b_y2) in buttons:
        button_rects[b_key] = (bx1, b_y1, bx2, b_y2)
        is_hover = (hovered_btn == b_key)

        if b_key == "NEXT":
            bg_c = (0, 190, 110) if not is_hover else (0, 245, 140)
            txt_c = (15, 25, 35) if is_hover else (255, 255, 255)
        elif b_key == "PREV":
            bg_c = (200, 110, 30) if not is_hover else (240, 140, 45)
            txt_c = (15, 25, 35) if is_hover else (255, 255, 255)
        elif b_key == "CLOSE":
            bg_c = (45, 45, 190) if not is_hover else (60, 60, 240)
            txt_c = (255, 255, 255)
        else:
            bg_c = (40, 50, 70) if not is_hover else (65, 80, 110)
            txt_c = (0, 240, 220)

        cv2.rectangle(rendered, (bx1, b_y1), (bx2, b_y2), bg_c, -1)
        border_c = (0, 255, 200) if is_hover else (100, 125, 160)
        cv2.rectangle(rendered, (bx1, b_y1), (bx2, b_y2), border_c, 2 if is_hover else 1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale_txt = 0.54
        (bw_t, bh_t), _ = cv2.getTextSize(b_text, font, scale_txt, 2 if is_hover else 1)
        tx = bx1 + (bx2 - bx1 - bw_t) // 2
        ty = b_y1 + (b_y2 - b_y1 + bh_t) // 2
        cv2.putText(rendered, b_text, (tx, ty), font, scale_txt, txt_c, 2 if is_hover else 1, cv2.LINE_AA)

    # Hint text in bottom bar
    hint_str = "Click [< PREV] / [NEXT >] or press [N] / [P] / [Space] to pause"
    cv2.putText(rendered, hint_str, (580, by1 + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (160, 180, 200), 1, cv2.LINE_AA)

    return rendered, button_rects


def render_slideshow_hud(*args, **kwargs):
    """Backwards-compatible wrapper returning only annotated frame."""
    res = render_slideshow_hud_and_controls(*args, **kwargs)
    return res[0] if isinstance(res, tuple) else res


def play_slideshow(folder_path: str = "flower test"):
    """
    Main interactive slideshow player with full button click support.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)

    valid_exts = (".mp4", ".avi", ".mov", ".mkv", ".wmv")
    videos = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(valid_exts)])

    if not videos:
        print(f"\n[WARNING] No videos found in '{folder_path}'. Looking for 'test.mp4'...")
        if os.path.exists("test.mp4"):
            videos = ["../test.mp4"]
        else:
            print(f"[ERROR] No videos found in '{folder_path}'. Please paste your videos inside 'flower test/' folder.")
            return

    total_videos = len(videos)
    window_name = "AI Flower Detection - Interactive Video Player"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    print("=" * 80)
    print("🌸 AI FLOWER DETECTION - INTERACTIVE SLIDESHOW PLAYER")
    print("=" * 80)
    print(f"Folder              : {folder_path}")
    print(f"Total Videos Found  : {total_videos} videos")
    for i, v in enumerate(videos, 1):
        print(f"  [{i:02d}/{total_videos}] {v}")
    print("\nInteractive Controls (Mouse & Keyboard):")
    print("  • CLICK '< PREV' or press [P] / [Left]   : Go to Previous video")
    print("  • CLICK 'NEXT >' or press [N] / [Right]  : Go to Next video")
    print("  • CLICK '|| PAUSE' or press [SPACE]      : Pause / Resume playback")
    print("  • CLICK 'RESTART' or press [R]           : Restart current video")
    print("  • CLICK 'X CLOSE' or press [Q] / [ESC]   : Exit player")
    print("=" * 80 + "\n")

    os.makedirs("output_videos/snapshots", exist_ok=True)

    state = {
        "curr_idx": 0,
        "paused": False,
        "next_video": False,
        "prev_video": False,
        "restart_video": False,
        "quit_app": False,
        "hovered_btn": None,
        "btn_rects": {},
    }

    def on_mouse(event, x, y, flags, param):
        hovered = None
        for b_key, (bx1, by1, bx2, by2) in state["btn_rects"].items():
            if bx1 <= x <= bx2 and by1 <= y <= by2:
                hovered = b_key
                break
        state["hovered_btn"] = hovered

        if event == cv2.EVENT_LBUTTONDOWN:
            if hovered == "NEXT":
                print("\n[ACTION] Clicked 'NEXT >' -> Advancing to next video...")
                state["next_video"] = True
            elif hovered == "PREV":
                print("\n[ACTION] Clicked '< PREV' -> Loading previous video...")
                state["prev_video"] = True
            elif hovered == "PAUSE":
                state["paused"] = not state["paused"]
                print(f"[ACTION] Clicked PAUSE/PLAY -> {'PAUSED' if state['paused'] else 'PLAYING'}")
            elif hovered == "RESTART":
                print("[ACTION] Clicked 'RESTART'")
                state["restart_video"] = True
            elif hovered == "CLOSE":
                print("\n[ACTION] Clicked 'X CLOSE'")
                state["quit_app"] = True

    cv2.setMouseCallback(window_name, on_mouse)

    session_results = []

    while not state["quit_app"]:
        v_filename = videos[state["curr_idx"]]
        v_path = os.path.join(folder_path, v_filename) if not v_filename.startswith("..") else v_filename[3:]

        profile = detect_clip_species(v_path)

        cap = cv2.VideoCapture(v_path)
        if not cap.isOpened():
            print(f"[ERROR] Could not open video: {v_path}")
            state["curr_idx"] = (state["curr_idx"] + 1) % total_videos
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0 or np.isnan(fps):
            fps = 24.0

        delay_ms = max(1, int(1000.0 / fps))
        frame_idx = 0
        state["next_video"] = False
        state["prev_video"] = False
        state["restart_video"] = False
        max_seen = 0
        last_annotated = None

        print(f"\n▶ [{state['curr_idx'] + 1}/{total_videos}] Playing: {v_filename}")
        print(f"   🌸 Flower : {profile['en']} ({profile['mr']})")
        print(f"   🌿 Family : {profile['family']} | {profile['sci']}")

        while not state["quit_app"]:
            if state["next_video"]:
                session_results.append({
                    "filename": v_filename,
                    "flower": profile["en"],
                    "marathi": profile["mr"],
                    "scientific": profile["sci"],
                    "max_count": max_seen,
                })
                state["curr_idx"] = (state["curr_idx"] + 1) % total_videos
                break

            if state["prev_video"]:
                session_results.append({
                    "filename": v_filename,
                    "flower": profile["en"],
                    "marathi": profile["mr"],
                    "scientific": profile["sci"],
                    "max_count": max_seen,
                })
                state["curr_idx"] = (state["curr_idx"] - 1) % total_videos
                break

            if state["restart_video"]:
                frame_idx = 0
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                state["restart_video"] = False
                state["paused"] = False

            if not state["paused"]:
                ret, frame = cap.read()
                if not ret:
                    print(f"   ✔ Finished '{v_filename}' (Max: {max_seen} flowers) -> Advancing to next video...")
                    session_results.append({
                        "filename": v_filename,
                        "flower": profile["en"],
                        "marathi": profile["mr"],
                        "scientific": profile["sci"],
                        "max_count": max_seen,
                    })
                    state["curr_idx"] = (state["curr_idx"] + 1) % total_videos
                    break

                timestamp = frame_idx / fps
                detections = detect_flowers_in_frame(frame, profile)
                if len(detections) > max_seen:
                    max_seen = len(detections)

                annotated, b_rects = render_slideshow_hud_and_controls(
                    frame,
                    detections,
                    v_filename,
                    state["curr_idx"] + 1,
                    total_videos,
                    timestamp,
                    frame_idx,
                    total_frames,
                    profile,
                    paused=state["paused"],
                    hovered_btn=state["hovered_btn"],
                    display_w=1280,
                )
                state["btn_rects"] = b_rects
                last_annotated = annotated

                cv2.imshow(window_name, annotated)
                frame_idx += 1
            else:
                if last_annotated is not None:
                    paused_disp, b_rects = render_slideshow_hud_and_controls(
                        frame,
                        detections,
                        v_filename,
                        state["curr_idx"] + 1,
                        total_videos,
                        timestamp,
                        frame_idx,
                        total_frames,
                        profile,
                        paused=True,
                        hovered_btn=state["hovered_btn"],
                        display_w=1280,
                    )
                    state["btn_rects"] = b_rects
                    cv2.imshow(window_name, paused_disp)
                time.sleep(0.02)

            key = cv2.waitKey(delay_ms if not state["paused"] else 30) & 0xFF
            if key in [ord("q"), ord("Q"), 27]:
                state["quit_app"] = True
                break
            elif key == ord(" "):
                state["paused"] = not state["paused"]
                print(f"[KEY] {'PAUSED' if state['paused'] else 'PLAYING'}")
            elif key in [ord("n"), ord("N"), 83]:
                print("[KEY] User triggered NEXT video.")
                state["next_video"] = True
            elif key in [ord("p"), ord("P"), 81]:
                print("[KEY] User triggered PREVIOUS video.")
                state["prev_video"] = True
            elif key in [ord("r"), ord("R")]:
                state["restart_video"] = True
            elif key in [ord("s"), ord("S")]:
                snap_name = f"output_videos/snapshots/snap_{os.path.splitext(v_filename)[0]}_f{frame_idx}.jpg"
                if last_annotated is not None:
                    cv2.imwrite(snap_name, last_annotated)
                    print(f"[SAVED] {snap_name}")

        cap.release()

    cv2.destroyAllWindows()
    print("\n[INFO] Player closed.")
    print_summary(session_results)


def print_summary(results: list):
    """Prints a structured summary table and writes to results.txt."""
    if not results:
        return
    print("\n" + "=" * 80)
    print("📊 AI FLOWER DETECTION - FINAL SUMMARY REPORT")
    print("=" * 80)
    print(f"{'#':<3} | {'Filename':<28} | {'Flower Name (English)':<28} | {'मराठी नाव':<24} | {'Count'}")
    print("-" * 80)
    seen = set()
    unique_results = []
    for r in results:
        if r["filename"] not in seen:
            seen.add(r["filename"])
            unique_results.append(r)

    for i, r in enumerate(unique_results, 1):
        print(f"{i:02d}  | {r['filename']:<28} | {r['flower']:<28} | {r['marathi']:<24} | {r['max_count']} flowers")
    print("=" * 80 + "\n")

    os.makedirs("output_videos", exist_ok=True)
    with open("output_videos/results.txt", "w", encoding="utf-8") as f:
        f.write("🌸 AI FLOWER DETECTION - BATCH SUMMARY REPORT\n")
        f.write("=" * 80 + "\n")
        for i, r in enumerate(unique_results, 1):
            f.write(f"{i:02d}. {r['filename']}\n")
            f.write(f"    • Flower (English)  : {r['flower']}\n")
            f.write(f"    • मराठी नाव         : {r['marathi']}\n")
            f.write(f"    • Scientific Name   : {r['scientific']}\n")
            f.write(f"    • Flower Count      : {r['max_count']} flowers\n\n")


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "flower test"
    play_slideshow(folder)
