#@markdown Landscape images and elemental functionality fully integrated; elements return correctly. family tree half-integrated

!pip install tabulate
!pip install beautifulsoup4 requests
!pip install astral
!pip install geopy
import warnings
import pytz
import uuid
import string
import os
import random
import string
import textwrap
import time
import torch
import ephem
import numpy as np
import requests
import json
import re
from IPython.display import Image as IPyImage
from PIL import ImageOps
from tzlocal import get_localzone
from bs4 import BeautifulSoup
from datetime import datetime
from astral.sun import sun
from astral import LocationInfo
from PIL import Image, ImageDraw, ImageFont
from tabulate import tabulate
from torchvision.transforms.functional import to_pil_image
#______Diffusion Req.s____
!pip install transformers
%pip install -qq git+https://github.com/ShivamShrirao/diffusers
%pip install -q -U --pre triton
%pip install -q accelerate transformers ftfy bitsandbytes==0.35.0 gradio natsort safetensors xformers
from transformers import AutoModel
from diffusers import DiffusionPipeline
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, DDIMScheduler

g_cuda = None
warnings.filterwarnings("ignore")

def save_game_state(dragons, filename="dragons.json"):
    dragon_data = [dragon.__dict__ for dragon in dragons]
    with open(filename, "w") as outfile:
        json.dump(dragon_data, outfile, indent=4, default=str)

def load_game_state(filename="dragons.json"):
    with open(filename, "r") as infile:
        dragon_data = json.load(infile)

    def convert_uuid(obj):
        if "id" in obj:
            obj["id"] = uuid.UUID(obj["id"])
        if "parent1_id" in obj and obj["parent1_id"] is not None:
            obj["parent1_id"] = uuid.UUID(obj["parent1_id"])
        if "parent2_id" in obj and obj["parent2_id"] is not None:
            obj["parent2_id"] = uuid.UUID(obj["parent2_id"])
        return obj

    dragons = [Dragon(**convert_uuid(dragon)) for dragon in dragon_data]
    return dragons


##---------------------------------------

class Dragon:
    def __init__(self, elements, appearance, base_stats, parents=None, life_stage="egg", hatch_time=10, grow_time=20,):
        self.id = uuid.uuid4()  # Assign a unique id to each dragon
        self.name = None
        self.elements = elements
        self.appearance = appearance
        self.base_stats = base_stats
        self.parents = parents
        self.life_stage = life_stage
        self.hatch_time = hatch_time
        self.grow_time = grow_time
        self.creation_time = time.time()
        self.offspring = []
        self.seed = None

        if parents is not None:
            parent1, parent2 = parents  # Get the parent dragon objects
            parent1_seed = parent1.seed if random.choice([True, False]) else random.randint(1, 4294967295)
            parent2_seed = parent2.seed if random.choice([True, False]) else random.randint(1, 4294967295)
            self.seed = hash(f"{parent1_seed}{parent2_seed}")
        else:
            self.seed = random.randint(1, 4294967295)

    def __str__(self):
        return dragon_description(self)

    def generate_seed(self, parent1_seed, parent2_seed):
        if parent1_seed <= parent2_seed:
            return hash(f"{parent1_seed}{parent2_seed}")
        else:
            return hash(f"{parent2_seed}{parent1_seed}")

        if parents is not None:
            self.parent1_id, self.parent2_id = parents
        else:
            self.parent1_id = None
            self.parent2_id = None


    def __str__(self):
        return dragon_description(self)
    
    def display(self):
        data = [
            ["Name", self.name],
            ["Color", self.appearance["color"]],
            ["Pattern", self.appearance["pattern"]],
            ["Active Element(s)", ", ".join(self.elements["active"])],
            ["Recessive Element", self.elements["recessive"]],
            ["Health", self.base_stats["health"]],
            ["Strength", self.base_stats["strength"]],
            ["Speed", self.base_stats["speed"]],
            ["Luck", self.base_stats["luck"]],
            ["Life Stage", self.life_stage]  # Display life stage
        ]

        return tabulate(data, tablefmt="grid")


    def hatch(self, name):
        if self.life_stage == "egg":
            if self.can_hatch():
                self.life_stage = "hatchling"
                self.name = name
                print(f"{self.name} has hatched into a hatchling!")
            else:
                remaining_time = self.hatch_time - (time.time() - self.creation_time)
                print(f"The egg needs {remaining_time:.2f} more seconds to hatch.")
        else:
            print("This dragon cannot hatch because it is not an egg.")

    def can_hatch(self):
        return (time.time() - self.creation_time) >= self.hatch_time

    def grow(self):
        if self.life_stage == "hatchling":
            if self.can_grow():
                self.life_stage = "adult"
                print(f"{self.name} has grown into an adult!")
            else:
                remaining_time = self.grow_time - (time.time() - self.creation_time)
                print(f"{self.name} needs {remaining_time:.2f} more seconds to grow.")
        else:
            print("This dragon cannot grow because it is not a hatchling.")

    def can_grow(self):
        return (time.time() - self.creation_time) >= self.grow_time

def generate_dragon_name():
    prefixes = [
        "Flame", "Fire", "Ice", "Frost", "Thunder", "Storm", "Earth", "Rock", 
        "Stone", "Wind", "Sky", "Rain", "Cloud", "Mist", "Shadow", "Night", 
        "Day", "Sun", "Moon", "Star", "Magic", "Mystic", "Crystal", "Metal",
        "Golden", "Silver", "Bronze", "Diamond", "Jade", "Ruby", "Sapphire",
    ]
    suffixes = [
        "wing", "claw", "tail", "scale", "tooth", "breath", "fire", "ice", 
        "thunder", "storm", "earth", "rock", "stone", "wind", "sky", "rain",
        "cloud", "mist", "shadow", "night", "day", "sun", "moon", "star",
        "magic", "mystic", "crystal", "metal", "beak", "feather", "horn",
    ]
    return random.choice(prefixes) + random.choice(suffixes)

##_________Element Inheriting Logic_______
def combine_primary_elements(element1, element2, element_interactions, parent1, parent2):
    # Handle the case when the elements are the same
    if element1 == element2:
        return element1, element1

    # Check if there is a hybrid element that can be created from the two primary elements
    hybrid_element = get_hybrid_element(element1, element2)
    if hybrid_element:
        # Check if either parent has a hybrid primary element
        if has_hybrid_primary(parent1) or has_hybrid_primary(parent2):
            print("Cannot breed dragons with hybrid primary elements.")
            return None

        # Roll a random number between 0 and 1, and check if it's less than or equal to the probability
        # of a hybrid element being selected as the primary element
        if random.random() <= 0.005:
            return hybrid_element, None

    # Check the interaction between the two primary elements
    if element1 in element_interactions[element2]["weak"]:
        return element1, element2
    elif element2 in element_interactions[element1]["weak"]:
        return element2, element1
    else:
        # Check if the parents' recessive primary elements match, their primary elements have no interaction,
        # and their active primary elements don't match
        recessive1 = parent1.elements['recessive_primary']
        recessive2 = parent2.elements['recessive_primary']
        if (recessive1 == recessive2 and element1 not in element_interactions[element2]["strong"] 
                and element2 not in element_interactions[element1]["strong"] and element1 != element2):
            primary = recessive1
            recessive = random.choice([element1, element2])
            return primary, recessive

        # If there is no interaction and the condition above is not met, choose one of the primary elements at random
        return random.choice([element1, element2]), element1

def combine_secondary_elements(element1, element2, element_interactions, parent1, parent2):
    # Handle the case when there is no secondary element
    if not element1 or not element2:
        return None, None

    # Handle the case when the elements are the same
    if element1 == element2:
        return element1, element1

    # Check the interaction between the two secondary elements
    if element1 in element_interactions[element2]["weak"]:
        return element1, element2
    elif element2 in element_interactions[element1]["weak"]:
        return element2, element1
    else:
        # If the parents have the same recessive secondary element, it becomes the recessive secondary element in the hybrid
        recessive1 = parent1.get("recessive_secondary")
        recessive2 = parent2.get("recessive_secondary")
        if recessive1 == recessive2:
            active_secondary = random.choice([element1, element2])
            recessive_secondary = recessive1
        else:
            # Choose one of the secondary elements at random to be the active secondary element
            # and the other to be the recessive secondary element
            active_secondary = random.choice([element1, element2])
            recessive_secondary = element1 if active_secondary == element2 else element2

        return active_secondary, recessive_secondary
def combine_recessive_elements(primary1, primary2, recessive1, recessive2):
    # Handle the case when both parents have the same primary element
    if primary1 == primary2:
        return recessive1, recessive2

    # If both parents have the same recessive element, it becomes a primary element in the hybrid
    if recessive1 == recessive2:
        return recessive1, None

    # If one parent has the recessive element and the other has the primary element, the recessive becomes
    # the secondary element in the hybrid
    if primary1 == recessive2:
        return recessive2, recessive1
    elif primary2 == recessive1:
        return recessive1, recessive2

    # If the parents have different recessive elements, choose one at random to be the recessive in the hybrid
    return random.choice([recessive1, recessive2]), None

def combine_elements(parent1_elements, parent2_elements, element_interactions):
    p1_primary = parent1_elements["primary"]
    p1_secondary = parent1_elements.get("secondary", None)
    p2_primary = parent2_elements["primary"]
    p2_secondary = parent2_elements.get("secondary", None)

    # Inherit primary element
    if p1_primary == p2_primary:
        primary = p1_primary
        recessive_primary = p2_primary
    elif p1_primary in element_interactions[p2_primary]["strong"]:
        primary = p2_primary
        recessive_primary = p1_primary
    elif p2_primary in element_interactions[p1_primary]["strong"]:
        primary = p1_primary
        recessive_primary = p2_primary
    else:
        primary = random.choice([p1_primary, p2_primary])
        recessive_primary = p1_primary if primary == p2_primary else p2_primary

    # Inherit secondary element
    if p1_secondary and p2_secondary:
        if p1_secondary == p2_secondary:
            secondary_recessive = p1_secondary
            secondary_active = p1_secondary
        elif p1_secondary in element_interactions[p2_secondary]["weak"]:
            secondary_active = p2_secondary
            secondary_recessive = p1_secondary
        elif p2_secondary in element_interactions[p1_secondary]["weak"]:
            secondary_active = p1_secondary
            secondary_recessive = p2_secondary
        else:
            secondary_active = random.choice([p1_secondary, p2_secondary])
            secondary_recessive = p1_secondary if secondary_active == p2_secondary else p2_secondary
    else:
        secondary_active = p1_secondary if p1_secondary else p2_secondary if p2_secondary else None
        secondary_recessive = None

    stat_boost = None  # Modify this based on your game logic if needed

    return primary, secondary_recessive, recessive_primary, secondary_active, stat_boost
def get_hybrid_element(element1, element2):
    hybrids = {
        frozenset(["fire", "water"]): "steam",
        frozenset(["fire", "earth"]): "lava",
        frozenset(["fire", "air"]): "smoke",
        frozenset(["water", "earth"]): "mud",
        frozenset(["water", "air"]): "mist",
        frozenset(["earth", "air"]): "sandstorm"
    }

    return hybrids.get(frozenset([element1, element2]))
##____Dragon Population______
def create_dragon_population(population_size):
    life_stages = ['egg', 'hatchling', 'adult']
    dragon_population = [random_dragon(life_stage=random.choice(life_stages)) for _ in range(population_size)]
    return dragon_population

def display_dragon_population(dragon_population):
    print("\nDragon Population:")
    for index, dragon in enumerate(dragon_population):
        life_stage = dragon.life_stage
        name = dragon.name or f"{life_stage.capitalize()} ({index})"
        print(f"{index}: {name} - Elements: {', '.join(dragon.elements.values())} - Life Stage: {life_stage.capitalize()}")

def find_mate(child_dragon, dragon_population):
    excluded_ids = {child_dragon.id, child_dragon.parent1_id, child_dragon.parent2_id}
    potential_mates = [dragon for dragon in dragon_population if dragon.id not in excluded_ids and dragon.life_stage == "adult"]

    if not potential_mates:
        return None

    return random.choice(potential_mates)

###_______Breeding__________
def breed_dragons(parent1, parent2, element_interactions):
    name = generate_dragon_name()
    primary_element, secondary_element, recessive_primary, recessive_secondary, stat_boost = combine_elements(parent1.elements, parent2.elements, element_interactions)

    elements = {
        "primary": primary_element,
        "secondary": secondary_element,
        "recessive_primary": recessive_primary,
        "recessive_secondary": recessive_secondary,
    }

    # Color inheritance
    primary_element = elements["primary"]
    if primary_element == parent1.elements["primary"]:
        color = parent1.appearance["color"]
    else:
        color = parent2.appearance["color"]

    appearance = {
        "color": color,
        "horn_shape": random.choice([parent1.appearance["horn_shape"], parent2.appearance["horn_shape"]]),
        "wing_shape": random.choice([parent1.appearance["wing_shape"], parent2.appearance["wing_shape"]]),
    }

    # Determine whether to apply a stat boost
    if (parent1.elements["secondary"], parent2.elements["secondary"]) in element_interactions.get("cancels_out", []):
        base_stats = {
            "health": random.randint(10, 100),
            "strength": random.randint(1, 50) + 5,
            "speed": random.randint(1, 50) + 5,
            "luck": random.randint(1, 10) + 1,
        }
    else:
        base_stats = {
            "health": random.randint(10, 100),
            "strength": random.randint(1, 50),
            "speed": random.randint(1, 50),
            "luck": random.randint(1, 10),
        }

    return Dragon(elements=elements, appearance=appearance, base_stats=base_stats, parents=(parent1, parent2))

element_color_ranges = {
    "fire": ["red", "orange", "yellow", "scarlet", "crimson"],
    "water": ["blue", "aqua", "turquoise", "teal", "navy"],
    "earth": ["brown", "green", "tan", "umber", "ochre"],
    "air": ["white", "sky blue", "light gray", "light pink", "pale yellow"],
    "astral": ["purple", "violet", "indigo", "amethyst", "lavender"],
    "life": ["green", "spring green", "emerald", "lime", "forest"],
    "death": ["black", "dark gray", "maroon", "charcoal", "dark purple"],
    "chaos": ["rainbow", "multicolored", "iridescent", "prismatic", "kaleidoscopic"],
    "order": ["sepia", "gray", "white", "platinum", "pewter"],
    "void": ["black", "midnight blue", "dark purple", "shadow", "obsidian"],
    "time": ["gold", "silver", "bronze", "copper", "metallic"],
}


def random_dragon(life_stage="adult"):
    random.seed()
    name = generate_dragon_name()

    primary_elements = ["fire", "water", "earth", "air"]
    secondary_elements = [
        ("astral", 0.3),
        ("life", 0.25),
        ("death", 0.2),
        ("chaos", 0.15),
        ("order", 0.1),
        ("void", 0.05),
        ("time", 0.03),
    ]

    random_primary = random.choice(primary_elements)
    random_secondary = random.choices(secondary_elements, weights=[p[1] for p in secondary_elements])[0][0]

    # Select two random recessive elements from the list of available elements
    all_elements = primary_elements + [e[0] for e in secondary_elements]
    random_recessives = random.sample(all_elements, k=2)

    elements = {
        "primary": random_primary,
        "secondary": random_secondary,
        "recessive_primary": random_recessives[0],
        "recessive_secondary": random_recessives[1]
    }
    horn_shapes = ["curved", "straight", "spiral", "forked", "wavy", "branching", "crescent-shaped", "bladelike"]
    wing_shapes = ["feathered", "leathery", "webbed", "angel", "mechanical", "insect-like", "crystal", "transparent", "ethereal", "petal-like"]

    appearance = {
        "color": element_color(elements, element_color_ranges),
        "horn_shape": random.choice(horn_shapes),
        "wing_shape": random.choice(wing_shapes),
    }
    base_stats = {
        "health": random.randint(10, 100),
        "strength": random.randint(1, 50),
        "speed": random.randint(1, 50),
        "luck": random.randint(1, 10),
    }

    dragon = Dragon(elements, appearance, base_stats, life_stage=life_stage)
    
    if life_stage != "egg":
        dragon.name = name
        if life_stage == "hatchling":
            dragon.creation_time -= dragon.hatch_time
        elif life_stage == "adult":
            dragon.creation_time -= (dragon.hatch_time + dragon.grow_time)

    return dragon


def element_color(elements, element_color_ranges):
    primary_element = elements["primary"]
    return random.choice(element_color_ranges[primary_element])

##___Environment/Landscape Setup____

element_weather_mapping = {
    "fire": {"weather_condition": ["clear sky", "volcanic ash", "smoke"]},
    "water": {
        "slightly increased": {"weather_condition": ["light rain", "few clouds", "light intensity drizzle", "light intensity drizzle rain", "drizzle rain", "light intensity shower rain", "snow", "sleet", "shower sleet", "squalls"]},
        "moderately increased": {"weather_condition": ["moderate rain", "drizzle", "heavy intensity drizzle", "heavy intensity drizzle rain", "shower rain and drizzle", "heavy shower rain and drizzle", "shower drizzle", "shower rain", "heavy snow", "light rain and snow", "light shower snow", "squalls"]},
        "greatly increased": {"weather_condition": ["heavy intensity rain", "very heavy rain", "extreme rain", "heavy intensity shower rain", "ragged shower rain", "heavy shower snow"]},
    },
    "earth": {
        "slightly increased": {"weather_condition": ["mist", "smoke", "haze", "dust/sand whirls", "sand", "dust", "volcanic ash", "dust/sand whirls"]},
        "moderately increased": {"weather_condition": ["fog"]},
        "greatly increased": {"weather_condition": ["tornado"]},
    },
    "air": {
        "slightly increased": {"weather_condition": ["light thunderstorm", "dust/sand whirls", "mist", "thunderstorm with light drizzle", "volcanic ash", "smoke"]},
        "moderately increased": {"weather_condition": ["thunderstorm with light rain", "thunderstorm with rain", "thunderstorm", "thunderstorm with heavy drizzle", "sand", "volcanic ash", "squalls"]},
        "greatly increased": {"weather_condition": ["thunderstorm with heavy rain", "heavy thunderstorm", "ragged thunderstorm", "tornado"]},
    },
    "death": {"weather_condition": ["tornado", "heavy intensity rain", "very heavy rain", "extreme rain", "heavy intensity shower rain", "ragged shower rain", "heavy shower snow"]},
    "chaos": {"weather_condition": ["tornado"]},
    "order": {"weather_condition": []},
    "time": {"weather_condition": []},
    "astral": {"weather_condition": []},
    "void": {"weather_condition": []},
    "light": {"weather_condition": []},
    "dark": {
        "slightly increased": {"weather_condition": ["overcast clouds"]},
    },
    "life": {
        "slightly increased": {"weather_condition": ["light rain"]},
    },
}



element_landscape_mapping = {
    "fire": {"features": ["volcano", "cave", "geyser"]},
    "water": {"features": ["river", "lake", "waterfall", "beach", "cove"]},
    "earth": {"features": ["mountain", "canyon", "hill", "cliff", "valley"]},
    "air": {"features": ["plateau", "crag", "meadow"]},
    "life": {"features": ["field", "forest", "marsh", "swamp"]},
    "death": {"features": ["tundra", "desert", "cave"]},
    "chaos": {"features": ["volcano", "geyser"]},
    "order": {"features": ["mountain", "plateau", "crag"]},
    "astral": {"features": ["canyon", "valley", "meadow"]},
    "void": {"features": ["cave", "crag", "hill"]},
    "time": {"features": [], "time_of_day": ["dawn", "noon", "dusk"], "current_moon_phase": ["New Moon", "First Quarter", "Full Moon", "Last Quarter"]},
    "dark": {"features": [], "time_of_day": ["night"], "current_moon_phase": ["Waning Crescent", "Waning Gibbous"]},
    "light": {"features": [], "time_of_day": ["morning", "late morning", "early afternoon", "afternoon"], "current_moon_phase": ["Waxing Crescent", "Waxing Gibbous"]}

}

adjectives = ['beautiful', 'majestic', 'serene', 'breathtaking', 'picturesque', 'stunning', 'idyllic', 'scenic', 'charming', 'enchanting', 'magnificent', 'spectacular', 'tranquil', 'peaceful', 'pristine', 'vibrant', 'glorious', 'splendid', 'dramatic', 'awe-inspiring', 'spellbinding', 'dreamy', 'fantastic', 'amazing', 'phenomenal', 'wondrous', 'miraculous', 'mystical']
features = ['field', 'mountain', 'river', 'lake', 'waterfall', 'cave', 'canyon', 'forest', 'meadow', 'ocean', 'beach', 'cove', 'valley', 'glacier', 'volcano', 'island', 'plateau', 'reef', 'crag', 'hill', 'cliff', 'cove', 'marsh', 'swamp', 'tundra', 'geyser', 'brook', 'creek']

feature = {"adjectives": adjectives, "features": features}

adjectives = feature["adjectives"]
features = feature["features"]

def indefinite_article(adjective):
    vowels = ['a', 'e', 'i', 'o', 'u']
    if adjective[0].lower() in vowels:
        return 'an'
    else:
        return 'a'
###______Time/Date/Weather/Location_________

def get_location():
    try:
        response = requests.get("https://ipinfo.io/json")
        location = response.json()
        return location['city'], location['region'], location['country']
    except Exception as e:
        print(f"Error retrieving location: {e}")
        return None

def get_weather_data(api_key, city, country):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city},{country}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return {
            "temperature": data["main"]["temp"],
            "weather_condition": data["weather"][0]["description"]
        }
    else:
        print(f"Error retrieving weather data: {response.status_code}")
        return None

def get_location_and_weather():
    try:
        response = requests.get("https://ipinfo.io/json")
        location = response.json()
        city, region, country = location["city"], location["region"], location["country"]

        api_key = "f934774accdc1c2584b5772ec6533d32"  # Replace with your own API key
        weather_data = get_weather_data(api_key, city, country)
        if weather_data:
            temperature, weather_condition = weather_data["temperature"], weather_data["weather_condition"]
        else:
            temperature, weather_condition = "Unknown", "Unknown"

        return city, region, country, temperature, weather_condition
    except Exception as e:
        print(f"Error retrieving location and weather data: {e}")
        return None

def time_to_time_of_day(local_time):
    hour = int(local_time.split(':')[0])
    
    if 4 <= hour < 6:
        return 'dawn'
    elif 6 <= hour < 9:
        return 'morning'
    elif 9 <= hour < 12:
        return 'late morning'
    elif 12 <= hour < 14:
        return 'noon'
    elif 14 <= hour < 16:
        return 'early afternoon'
    elif 16 <= hour < 18:
        return 'afternoon'
    elif 18 <= hour < 21:
        return 'evening'
    else:
        return 'night'


def get_moon_phase(date=None):
    if date is None:
        date = ephem.now()
    moon = ephem.Moon(date)
    sun = ephem.Sun(date)
    moon.compute(date)
    sun.compute(date)
    elongation = ephem.separation(moon, sun)
    phase = (1 + ephem.cos(elongation)) / 2
    phase_name = None
    if phase <= 0.0625 or phase > 0.9375:
        phase_name = "New Moon"
    elif 0.0625 < phase <= 0.1875:
        phase_name = "Waxing Crescent"
    elif 0.1875 < phase <= 0.3125:
        phase_name = "First Quarter"
    elif 0.3125 < phase <= 0.4375:
        phase_name = "Waxing Gibbous"
    elif 0.4375 < phase <= 0.5625:
        phase_name = "Full Moon"
    elif 0.5625 < phase <= 0.6875:
        phase_name = "Waning Gibbous"
    elif 0.6875 < phase <= 0.8125:
        phase_name = "Last Quarter"
    elif 0.8125 < phase <= 0.9375:
        phase_name = "Waning Crescent"
    return phase_name

def celsius_to_fahrenheit(celsius):
    return (celsius * 9/5) + 32

def landscape_prompt_for_elements():
    city, region, country = get_location()
    api_key = "f934774accdc1c2584b5772ec6533d32"  # Replace with your own API key
    weather_data = get_weather_data(api_key, city, country)

    if weather_data:
        temperature, weather_condition = weather_data["temperature"], weather_data["weather_condition"]
    else:
        temperature, weather_condition = "Unknown", "Unknown"

    local_tz = get_localzone()
    local_time = datetime.now(local_tz).strftime("%H:%M")

    time_of_day = time_to_time_of_day(local_time)

    feature = random.choice(features)

    return city, region, country, temperature, weather_condition, time_of_day, feature


def get_environmental_elements(city, region, country, temperature, weather_condition, current_moon_phase, local_time, time_of_day, feature):
    # Modify the fire element mapping based on the time of day
    if time_of_day != 'night':
        element_weather_mapping['fire']['weather_condition'] = ["clear sky", "volcanic ash", "smoke"]
    else:
        element_weather_mapping['fire']['weather_condition'] = ["volcanic ash", "smoke"]

    elements_in_play = []

    for element, mapping in element_landscape_mapping.items():
        conditions_met = []

        if "current_moon_phase" in mapping:
            if current_moon_phase in mapping["current_moon_phase"]:
                conditions_met.append(True)
            else:
                conditions_met.append(False)

        if "time_of_day" in mapping:
            if time_of_day in mapping["time_of_day"]:
                conditions_met.append(True)
            else:
                conditions_met.append(False)

        if "features" in mapping:
            if feature in mapping["features"]:
                conditions_met.append(True)
            else:
                conditions_met.append(False)

        if "temperature" in mapping:
            min_temp = mapping["temperature"].get("min", float('-inf'))
            max_temp = mapping["temperature"].get("max", float('inf'))
            if min_temp <= temperature_celsius <= max_temp:
                conditions_met.append(True)
            else:
                conditions_met.append(False)

        if "weather_condition" in element_weather_mapping[element]:
            if weather_condition in element_weather_mapping[element]["weather_condition"]:
                conditions_met.append(True)
            else:
                conditions_met.append(False)

        if any(conditions_met):
            elements_in_play.append(element)

    return elements_in_play

def display_current_conditions(city, region, country, temperature, weather_condition, time_of_day, feature):
    local_tz = get_localzone()
    local_time = datetime.now(local_tz).strftime("%H:%M")
    local_date = datetime.now(local_tz).strftime("%Y-%m-%d")
    current_moon_phase = get_moon_phase(ephem.now())

    if temperature != "Unknown":
        temperature_celsius = float(temperature)
        temperature_fahrenheit = celsius_to_fahrenheit(temperature_celsius)
    else:
        temperature_celsius, temperature_fahrenheit = "Unknown", "Unknown"

    elements_in_play = get_environmental_elements(city, region, country, temperature_celsius, weather_condition, current_moon_phase, local_time, time_of_day, feature)

    print(f"Current conditions in {city}, {region}, {country}:")
    print(f"Weather: {weather_condition}")
    print(f"Temperature: {temperature_celsius}°C / {temperature_fahrenheit}°F")
    print(f"Date: {local_date}, Time: {local_time}")
    print(f"Moon Phase: {current_moon_phase}")

    # Print the elements in play
    if elements_in_play:
        print("Elements in play:")
        for element in elements_in_play:
            print(f"- {element}")
    else:
        print("No elements in play.")
    if elements_in_play:
        print("Elements in play with stat boosts:")
        for element in elements_in_play:
            stat_boost = 0
            for increase_level, mapping in element_weather_mapping[element].items():
                if weather_condition in mapping["weather_condition"]:
                    if increase_level == "slightly increased":
                        stat_boost = 2.5
                    elif increase_level == "moderately increased":
                        stat_boost = 5
                    elif increase_level == "greatly increased":
                        stat_boost = 10
            print(f"- {element}: {stat_boost}%")
    else:
        print("No elements in play with stat boosts.")
####___Dragon/Env Prompts_____
def dragon_egg_prompt(dragon):
    name = dragon.name
    elements = dragon.elements
    appearance = dragon.appearance
    primary_element = elements['primary']
    secondary_element = elements['secondary']
    secondary_element_color = element_color_ranges.get(dragon.elements.get('secondary', 'void'), [])[0]
    description = (
        f"a {appearance['color']} and {secondary_element_color} {primary_element} {secondary_element} "
        f"dragon egg xcnvmbbx"
    )

    return description

def hatchling_dragon_prompt(dragon):
    name = dragon.name
    elements = dragon.elements
    appearance = dragon.appearance
    primary_element = elements['primary']
    secondary_element = elements['secondary']
    secondary_element_color = element_color_ranges.get(dragon.elements.get('secondary', 'void'), [])[0]
    description = (
        f"a cute newborn baby hatchling {appearance['color']} and {secondary_element_color} {primary_element} {secondary_element} "
        f"dragon ttdstdstfg"
    )

    return description

def adult_dragon_prompt(dragon):
    name = dragon.name
    elements = dragon.elements
    appearance = dragon.appearance
    primary_element = elements['primary']
    secondary_element = elements['secondary']
    secondary_element_color = element_color_ranges.get(dragon.elements.get('secondary', 'void'), [])[0]
    description = (
        f"a cute {appearance['color']} and {secondary_element_color} {primary_element} {secondary_element} "
        f"dragon with {appearance['horn_shape']} horns and {appearance['wing_shape']} wings asdffgwerte"
    )

    return description

def landscape_prompt():
    city, region, country = get_location()
    api_key = "f934774accdc1c2584b5772ec6533d32"  # Replace with your own API key
    weather_data = get_weather_data(api_key, city, country)

    if weather_data:
        temperature, weather_condition = weather_data["temperature"], weather_data["weather_condition"]
    else:
        temperature, weather_condition = "Unknown", "Unknown"

    local_tz = get_localzone()
    local_time = datetime.now(local_tz).strftime("%H:%M")

    time_of_day = time_to_time_of_day(local_time)

    adjective = random.choice(adjectives)
    feature = random.choice(features)
    article = indefinite_article(adjective)

    description = (
        f"{article} {adjective} {feature} in the {weather_condition} modern {city} {region} at {time_of_day} tzmzmmxznx"
    )
    return description

def generate_image(dragon, life_stage):#generate dragon images based on life stage
    if life_stage == 'egg':
        prompt = dragon_egg_prompt(dragon)
        model_config = model_configs[life_stage]
    elif life_stage == 'hatchling':
        prompt = hatchling_dragon_prompt(dragon)
        model_config = model_configs[life_stage]
    elif life_stage == 'adult':
        prompt = adult_dragon_prompt(dragon)
        model_config = model_configs[life_stage]
    else:
        raise ValueError(f"Invalid life stage: {life_stage}")

    pipe = StableDiffusionPipeline.from_pretrained(model_config['model_path'], safety_checker=None, torch_dtype=torch.float16).to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()

    g_cuda = torch.Generator(device='cuda')
    seed = dragon.seed
    g_cuda.manual_seed(seed)
    bit_depth_output = 32
    num_samples = 1
    height = 512
    width = 512

    with autocast("cuda"), torch.inference_mode():
        images = pipe(
            prompt,
            height=height,
            width=width,
            negative_prompt=model_config['negative_prompt'],
            num_images_per_prompt=num_samples,
            num_inference_steps=model_config['num_inference_steps'],
            guidance_scale=model_config['guidance_scale'],
            generator=g_cuda
        ).images

    # Return the generated image
        return images[0]

def generate_landscape():
    prompt = landscape_prompt()
    model_config = model_configs['landscape']

    # Load the model
    pipe = StableDiffusionPipeline.from_pretrained(model_config['model_path'], safety_checker=None, torch_dtype=torch.float16).to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()

    g_cuda = torch.Generator(device='cuda')
    seed = torch.randint(0, 2**32, (1,)).item()  # Random seed for landscape generation
    g_cuda.manual_seed(seed)
    bit_depth_output = 32
    num_samples = 1
    height = 512
    width = 512

    with autocast("cuda"), torch.inference_mode():
        images = pipe(
            prompt,
            height=height,
            width=width,
            negative_prompt=model_config['negative_prompt'],
            num_images_per_prompt=num_samples,
            num_inference_steps=model_config['num_inference_steps'],
            guidance_scale=model_config['guidance_scale'],
            generator=g_cuda
        ).images

    # Return the generated image
    return images[0]
   
model_configs = {
    'egg': {
        'model_path': "KaliYuga/dragon_egg",
        'negative_prompt': 'Ugly, deformed, noisy, blurry, distorted, grainy',
        'guidance_scale': 4.5,
        'num_inference_steps': 30
    },
    'hatchling': {
        'model_path': "KaliYuga/hatchling",
        'negative_prompt': 'Ugly, deformed, noisy, blurry, distorted, grainy',
        'guidance_scale': 4.5,
        'num_inference_steps': 35
    },
    'adult': {
        'model_path': "KaliYuga/adult_dragon",
        'negative_prompt': 'Ugly, deformed, noisy, blurry, distorted, grainy',
        'guidance_scale': 4.5,
        'num_inference_steps': 40
    },
    'landscape': {
        'model_path': "KaliYuga/landscape",
        'negative_prompt': 'Ugly, deformed, noisy, blurry, distorted, grainy',
        'guidance_scale': 4.5,
        'num_inference_steps': 30
    }
}
##_______Description for Card________
def dragon_description(dragon):
    name = dragon.name
    elements = dragon.elements
    appearance = dragon.appearance
    secondary_element_color = element_color_ranges.get(dragon.elements.get('secondary', 'void'), [])[0]

    if dragon.name is None:
        return f"This is a {appearance['color']} and {secondary_element_color} dragon egg."
    elif dragon.life_stage == "hatchling":
        return f"{name} is a {appearance['color']} and {secondary_element_color} hatchling."
    else:
        description = (
            f"{name} is a {appearance['color']} and {secondary_element_color} dragon "
            f"with {appearance['horn_shape']} horns and {appearance['wing_shape']} wings."
        )
        return description

def inherit_attribute(attr1, attr2):
    child_attr = {}
    for key in attr1:
        if isinstance(attr1[key], int) and isinstance(attr2[key], int):
            base_value = (attr1[key] + attr2[key]) // 2
            variation = random.randint(-5, 5)  # Adjust the range for more or less variation
            child_attr[key] = max(base_value + variation, 1)  # Ensure the value stays positive
        else:
            child_attr[key] = random.choice([attr1[key], attr2[key]])
    return child_attr

def create_trading_card(dragon):
    font_path = "/content/drive/MyDrive/pfeffer-mediaeval/PfefferMediaeval.otf"
    width, height = 500, 666
    card = Image.new('RGBA', (width, height), 'tan')

    draw = ImageDraw.Draw(card)
    border_color = 'black'
    border_thickness = 3
    draw.rectangle([(0, 0), (width - 1, height - 1)], outline=border_color, width=border_thickness)

    font_size = 20
    font = ImageFont.truetype(font_path, size=font_size)

    description = dragon_description(dragon)

    if dragon.life_stage == "egg":
        title = "Dragon Egg"
        image = generate_image(dragon, "egg")
    elif dragon.life_stage == "hatchling":
        title = f"{dragon.name} (Hatchling)"
        image = generate_image(dragon, "hatchling")
    else:
        title = dragon.name
        image = generate_image(dragon, "adult")

    title_width, title_height = draw.textsize(title, font=font)
    draw.text(((width - title_width) // 2, 10), title, font=font, fill='black')

    # Add the dragon image to the trading card
    image = Image.fromarray(np.uint8(image)).convert("RGBA")
    image = image.resize((300, 300))

    # Add a border around the dragon image
    image_border_thickness = 3
    image_border_color = 'black'
    image_border = Image.new('RGBA', (image.size[0] + 2*image_border_thickness, image.size[1] + 2*image_border_thickness), image_border_color)
    image_border.paste(image, (image_border_thickness, image_border_thickness))

    card.paste(image_border, ((width - image_border.size[0])//2, title_height+20))

    # Add the dragon attributes to the trading card
    y_offset = title_height + image_border.size[1] + 50
    elements_str = f"Elements: {dragon.elements['primary']} and {dragon.elements.get('secondary', '-')}"
    draw.text((10, y_offset), elements_str, font=font, fill='black')
    y_offset += font_size + 10

    if dragon.life_stage != "egg":  # Only show stats for hatched dragons
        for stat, value in dragon.base_stats.items():
            draw.text((10, y_offset), f"{stat}: {value}", font=font, fill='black')
            y_offset += font_size + 10

    # Add the dragon description to the trading card with text wrapping
    wrap_width = 50
    wrapped_description = textwrap.fill(description, width=wrap_width) 
    description_width, description_height = draw.textsize(wrapped_description, font=font)
    draw.text(((width - description_width) // 2, y_offset), wrapped_description, font=font, fill='black')

    file_name = f"{title}_trading_card.png"
    card.save(file_name)
    card.show()
    img = Image.fromarray(np.array(card).astype(np.uint8))
    return file_name

###____Family Tree Card____    
def create_family_tree_card(dragon, max_generations=3):
    config = {
        "font_path": "/content/drive/MyDrive/pfeffer-mediaeval/PfefferMediaeval.otf",
        "card_width": 1200,
        "card_height": 800,
        "image_width": 200,
        "image_height": 200,
        "border_thickness": 3,
        "font_size": 20,
        "border_color": 'black',
    }
    config["font"] = ImageFont.truetype(config["font_path"], size=config["font_size"])

    # Initialize the card with a tan background and a black border
    card = Image.new('RGBA', (config["card_width"], config["card_height"]), 'tan')
    draw = ImageDraw.Draw(card)
    draw.rectangle([(0, 0), (config["card_width"] - 1, config["card_height"] - 1)], outline=config["border_color"], width=config["border_thickness"])

    # Set up a dictionary to keep track of the coordinates of each dragon image on the card
    coordinates = {}

    # Add the starting dragon's name to the card
    dragon_name = dragon.name if dragon.name is not None else "Unnamed Dragon"
    title_width, title_height = draw.textsize(dragon_name, font=config["font"])
    draw.text(((config["card_width"] - title_width) // 2, config["image_height"] + 2 * config["border_thickness"]), dragon_name, font=config["font"], fill='black')

    # Add the starting dragon's elements to the card
    elements_str = f"Elements: {dragon.elements['primary']} and {dragon.elements.get('secondary', '-')}"
    elements_width, elements_height = draw.textsize(elements_str, font=config["font"])
    draw.text(((config["card_width"] - elements_width) // 2, config["image_height"] + title_height + 3 * config["border_thickness"]), elements_str, font=config["font"], fill='black')

    # Add the starting dragon's primary and secondary recessive elements to the card
    primary_recessive_str = f"Primary recessive: {dragon.elements['recessive_primary']}"
    primary_recessive_width, primary_recessive_height = draw.textsize(primary_recessive_str, font=config["font"])
    draw.text(((config["card_width"] - primary_recessive_width) // 2, config["image_height"] + title_height + elements_height + 4 * config["border_thickness"]), primary_recessive_str, font=config["font"], fill='black')

    if dragon.elements.get('secondary_recessive') is not None:
        secondary_recessive_str = f"Secondary recessive: {dragon.elements['recessive_secondary']}"
        secondary_recessive_width, secondary_recessive_height = draw.textsize(secondary_recessive_str, font=config["font"])
        draw.text(((config["card_width"] - secondary_recessive_width) // 2, config["image_height"] + title_height + elements_height + primary_recessive_height + 5 * config["border_thickness"]), secondary_recessive_str, font=config["font"], fill='black')

    # Recursively add ancestors to the card up to the specified number of generations
    add_ancestors_to_card(dragon, card, coordinates, config, max_generations=max_generations, x_offset=config["image_width"] // 2, y_offset=title_height + elements_height)
    # Save the family tree card to a file and display it on the screen
    file_name = f"{dragon.name}_family_tree.png"
    card.save(file_name)
    card.show()
    img = Image.fromarray(np.array(card).astype(np.uint8))
    return file_name

def add_ancestors_to_card(dragon, card, coordinates, config, generation=0, max_generations=3, x_offset=0, y_offset=0):

    if generation >= max_generations:
        return

    # Recursively add the dragon's parents to the card
    if dragon.parents is not None:
        for parent in dragon.parents:
            parent_id = parent.id
            if parent_id in coordinates:
                # If the parent is already on the card, just draw a line to it
                parent_x, parent_y = coordinates[parent_id]
                child_x = parent_x + x_offset
                child_y = parent_y + config["image_height"] + y_offset + 2 * config["font_size"]
                draw = ImageDraw.Draw(card)
                draw.line([(child_x, child_y), (parent_x + config["image_width"] // 2, parent_y + config["image_height"])], width=config["border_thickness"], fill=config["border_color"])
            else:
                # If the parent is not on the card yet, add it to the card
                parent_image = generate_image(parent, "adult")
                parent_image = Image.fromarray(np.uint8(parent_image)).convert("RGBA")
                parent_image = parent_image.resize((config["image_width"], config["image_height"]))
                parent_x = (config["card_width"] - (2 ** (max_generations-generation) * config["image_width"] + (2 ** (max_generations-generation-1)) * x_offset)) // 2
                parent_y = y_offset + (generation + 1) * (config["image_height"] + 2 * config["font_size"] + config["border_thickness"])

                card.paste(parent_image, (parent_x, parent_y), parent_image)
                coordinates[parent_id] = (parent_x, parent_y)

                # Add the parent's name to the card
                parent_name = parent.name if parent.name is not None else "Unnamed Dragon"
                draw = ImageDraw.Draw(card)
                name_width, name_height = draw.textsize(parent_name, font=config["font"])
                draw.text((parent_x + (config["image_width"] - name_width) // 2, parent_y - name_height - config["border_thickness"]), parent_name, font=config["font"], fill='black')

                # Draw a line from the child to the parent
                child_x = parent_x + x_offset
                child_y = parent_y + config["image_height"] + y_offset + 2 * config["font_size"]
                draw.line([(child_x, child_y), (parent_x + config["image_width"] // 2, parent_y)], width=config["border_thickness"], fill=config["border_color"])

                # Continue with the next generation
                add_ancestors_to_card(parent, card, coordinates, config, generation=generation + 1, max_generations=max_generations, x_offset=config["image_width"] // (2 ** (generation + 1)), y_offset=y_offset)
    
####____MAIN________
def main():
    city, region, country, temperature, weather_condition, time_of_day, feature = landscape_prompt_for_elements()
    display_current_conditions(city, region, country, temperature, weather_condition, time_of_day, feature)

    # Initialize the dragon population
    dragon_population = create_dragon_population(10)
  
    element_interactions = {
        "fire": {"weak": ["water"], "strong": ["air"]},
        "water": {"weak": ["earth"], "strong": ["fire"]},
        "earth": {"weak": ["air"], "strong": ["water"]},
        "air": {"weak": ["fire"], "strong": ["earth"]},
        "life": {"strong": ["death"], "weak": ["chaos"]},
        "death": {"strong": ["chaos"], "weak": ["order"]},
        "chaos": {"strong": ["order"], "weak": ["time"]},
        "order": {"strong": ["time"], "weak": ["life"]},
        "astral": {"strong": ["void"], "weak": ["order"]},
        "void": {"strong": ["astral"], "weak": ["chaos"]},
        "time": {"strong": ["life"], "weak": ["death"]}
    }

    try:
        dragons = load_game_state()
        print("Loaded previous game state.")
    except FileNotFoundError:
        print("No saved game found. Generating new dragon.")
        dragons = [random_dragon(life_stage="adult")]
        dragon_population.append(dragons[-1])
    last_original_dragons_indices = [len(dragons) - 1]



    while True:

        if len(dragons) == 1:
            dragon = dragons[-1]
            create_trading_card(dragon)
        elif len(dragons) > 2:
            dragon = dragons[-1]
            create_trading_card(dragon)

        print("\nOptions:")
        print("1. Save current game")
        print("2. Load saved game")
        print("3. View Surroundings")
        print("4. Generate new dragons")
        print("5. Display dragon population")
        print("6. Create a new dragon")
        print("7. Find a mate for a dragon and create offspring")
        print("8. Show family tree")
        print("9. Hatch an egg/Grow a Hatchling")
        print("10. Exit")

        choice = input("Enter your choice (1-11): ")

        if choice == "1":
            save_game_state(dragons)
            print("Game saved successfully!")
            print('-' * 100) # Add a line of hyphens
        elif choice == "2":
            dragons = load_game_state()
            last_original_dragons_indices = [len(dragons) - 2, len(dragons) - 1]
            print("Loaded saved game.")
            print('-' * 100) # Add a line of hyphens
        elif choice == "3":
            landscape_image = generate_landscape()
            display(landscape_image)
            print(landscape_prompt()[:-10])
            print('-' * 100) # Add a line of hyphens
        elif choice == "4":
            dragons.extend([random_dragon() for _ in range(2)])
            last_original_dragons_indices = [len(dragons) - 2, len(dragons) - 1]
            print('-' * 100) # Add a line of hyphens
        elif choice == "5":
            for idx, dragon in enumerate(dragon_population):
                print(f"{idx}: {dragon.name} - {dragon_description(dragon)}")
                print('-' * 100) # Add a line of hyphens
        elif choice == "6":
            new_dragon = random_dragon(life_stage="adult")
            dragon_population.append(new_dragon)
            print(f"Created new dragon: {new_dragon.name}")
            print('-' * 100) # Add a line of hyphens
        elif choice == "7":
            active_dragon = dragons[-1]
            mate = random_dragon(life_stage="adult")
            if active_dragon.life_stage != "adult":
                print("\nYou can only breed adult dragons.")
                continue
            print("\nParent 1:")
            create_trading_card(active_dragon)
            print("\nParent 2:")
            create_trading_card(mate)
            if mate.life_stage != "adult":
                print("\nYou can only breed adult dragons.")
                continue
            offspring = breed_dragons(active_dragon, mate, element_interactions)
            offspring.id = uuid.uuid4()
            print("\nOffspring:")
            create_trading_card(offspring)
            print("\nElemental Interactions:")
            print(f"{active_dragon.elements['primary']} vs {mate.elements['primary']}: {offspring.elements['primary']} inherited as primary element")
            if active_dragon.elements['secondary'] is not None and mate.elements['secondary'] is not None:
                print(f"{active_dragon.elements['secondary']} vs {mate.elements['secondary']}: {offspring.elements['secondary']} inherited as secondary element")
            print(f"Primary element recessive: {offspring.elements['recessive_primary']}")
            if offspring.elements['secondary'] is not None:
                print(f"Secondary element recessive: {offspring.elements['recessive_secondary']}")
            print(f"\nA new dragon has been added to your collection!")
            print('-' * 100) # Add a line of hyphens
            active_dragon.offspring.append((offspring.id, offspring))
            mate.offspring.append((offspring.id, offspring))
            dragons.append(offspring)
            dragon_population.extend([mate, offspring])
        elif choice == "8":
            if len(dragon_population) < 2:
                print("Not enough dragons in the population to create a family tree.")
                print('-' * 100) # Add a line of hyphens
            else:
                try:
                    dragon_index = int(input("Enter the index of the dragon to show family tree: "))
                    dragon = dragon_population[dragon_index]
                    print_family_tree(dragon)
                except (ValueError, IndexError):
                    print("Invalid index. Please try again.")
                    print('-' * 100) # Add a line of hyphens
        elif choice == "9":
            dragon_index = int(input("Enter the index of the dragon to hatch/grow: "))
            dragon = dragon_population[dragon_index]
            if dragon.life_stage == "egg":
                dragon_name = input("Enter a name for the hatchling: ")
                print("Egg:")
                create_trading_card(dragon)
                dragon.hatch(dragon_name)
                print("Hatchling:")
            elif dragon.life_stage == "hatchling":
                print("Hatchling:")
            else:
                print("Dragon is already an adult.")
                print('-' * 100) # Add a line of hyphens 
                continue
            create_trading_card(dragon)
            dragon.grow()
            print("Adult dragon:")
            create_trading_card(dragon)
            print('-' * 100) # Add a line of hyphens  
        elif choice == "10":
            break
        else:
            print("Invalid choice. Please try again.")
            print('-' * 100) # Add a line of hyphens

if __name__ == "__main__":
    main()            
