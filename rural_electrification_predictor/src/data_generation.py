"""
File: data_generation.py
Purpose: Generate a synthetic Bangalore rural electrification dataset.
Author: Your Name
Date: February 26, 2026
"""

import random
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def _build_village_names() -> List[str]:
    """Return the user-provided list of village names for dataset generation."""
    return [
        "Adihosahalli",
        "Agalakuppa",
        "Agasarahalli",
        "Ahobalapalya",
        "Aladahalli",
        "Alunaikanahalli",
        "Ananthapura",
        "Ananthapura",
        "Appagondanahalli",
        "Aralasandra",
        "Araledibba",
        "Arasinakunte (CT)",
        "Arebommanahalli",
        "Arjunabettahalli",
        "Avalakuppe",
        "Avverahalli",
        "Balaguruvanapalya",
        "Ballagere",
        "Banasavadi",
        "Baragoor",
        "Bardi",
        "Bargenahalli",
        "Basavanahalli",
        "Basavapatna",
        "Benchanahalli",
        "Bennegere",
        "Bettadahosahalli",
        "Bettahalli",
        "Bharathipura",
        "Bhattarahalli",
        "Bhavikere",
        "Bhoosandra",
        "Bidalur",
        "Bidanapalya",
        "Billanakote",
        "Biragondanahalli",
        "Biragundanahalli",
        "Bolamaranahalli",
        "Bolamaranahalli",
        "Bommanahalli",
        "Budihal",
        "Bugadihalli",
        "Byadarahalli",
        "Byadarahalli",
        "Byranahalli",
        "Byranaikanahalli",
        "Byrasandra",
        "Byrasettihalli",
        "Chandanahosahalli",
        "Channohalli",
        "Chikkanahalli",
        "Chikkanapalya",
        "Chikkannanahalli",
        "Chikkannanne",
        "Chikmaranahalli",
        "Chowdasandra",
        "Dasanapura",
        "Dasenahalli",
        "Deganahalli",
        "Devarahosahalli",
        "Doddabele",
        "Doddachannohalli",
        "Doddakarenahalli",
        "Dodderi",
        "Gangadharanapalya",
        "Gangenapura",
        "Gantehosahalli",
        "Geddalahalli",
        "Geddalahalli",
        "Ghandragulupura",
        "Giriyanapalya",
        "Gollahalli",
        "Goraghatta",
        "Goravanahalli",
        "Gorinabele",
        "Gottikere",
        "Govenahalli",
        "Govindapura",
        "Govindapura",
        "Gowrapura",
        "Guddegowdana Channohalli",
        "Gulapura",
        "Gundenahalli",
        "Gundenahalli",
        "Guruvanahalli",
        "H.Kempalinganahalli",
        "Hajipalya",
        "Halenahalli",
        "Halenijagal",
        "Halkur",
        "Hallurampura",
        "Hanchipura",
        "Hanumanapalya",
        "Hanumanthapura",
        "Harvesandra",
        "Hasurahalli",
        "Heggunda",
        "Honnarayanahalli",
        "Honnasandra",
        "Honnenahalli",
        "Hosahalli",
        "Hosahalli",
        "Hulleharve",
        "Huralihalli",
        "Hydalu",
        "Imchanahalli",
        "Iregenahalli",
        "Isuvanahalli",
        "Jajoor",
        "Jakkanahalli",
        "Jakkasandra",
        "Jakkasandra",
        "K.G. Srinivasapura",
        "K.Kempalinganahalli",
        "Kachanahalli",
        "Kadukarenahalli",
        "Kalalghatta",
        "Kallunaikanahalli",
        "Kamalapura",
        "Kambal",
        "Kannohalli",
        "Kanugondanahalli",
        "Kanuvanahalli",
        "Karehalli",
        "Karenahalli",
        "Karimanne",
        "Karimaranahalli",
        "Kasaragatta",
        "Kempapura Agrahara",
        "Kempohalli",
        "Kenchanahalli",
        "Kenchanapura",
        "Kengal",
        "Kengalkempohalli",
        "Kerekattiganoor",
        "Kodappanahalli",
        "Kodigebommanahalli",
        "Kodigehalli",
        "Kodihalli",
        "Kodipalya",
        "Koneripura",
        "Koolipura",
        "Koratagere",
        "Kothanahalli",
        "Krishnarajapalya",
        "Krishnarajapalya",
        "Krishnarajapura",
        "Krishnarajapura",
        "Krishnarajapura",
        "Kuluvanahalli",
        "Kuntbommanahalli",
        "Kuruvel Thimmanahalli",
        "Kuthagatta",
        "Lakkappanahalli",
        "Lakkasandra",
        "Lakkenahalli",
        "Lakkuru",
        "Lakshmanapura",
        "Lingenahalli",
        "Machenahalli",
        "Machonaikanahalli",
        "Madaga",
        "Madalakote",
        "Maddenahalli",
        "Maddenahalli",
        "Mahadevapura",
        "Mahimapura",
        "Makanakuppe",
        "Makenahalli",
        "Mallapura",
        "Mallarabanavadi",
        "Malonagathihalli",
        "Manchenahalli",
        "Mandigere",
        "Manne",
        "Mannerampura",
        "Mantanakurchi",
        "Maragondanahalli",
        "Maralakunte",
        "Marohalli",
        "Mavinakommenahalli",
        "Mavinakunte",
        "Melekattiganoor",
        "Minnapura",
        "Muddalinganahalli",
        "Mylanahalli",
        "Mylanahalli",
        "Narasapura",
        "Narasimanapalya",
        "Narasimhapalya",
        "Narasipalya",
        "Narasipura",
        "Narayanapura",
        "Narayanaraopalya",
        "Nelamangala (Rural)",
        "Nelamangala (TMC)",
        "Nidavanda",
        "Nijagal",
        "Nijagal Kempohalli",
        "Nimbenahalli",
        "Obalapura",
        "Obanaikanahalli",
        "Pallarahalli",
        "Pemmanahalli",
        "Ramanahalli",
    ]


def generate_dataset(output_path: str) -> pd.DataFrame:
    """
    Generate and save a synthetic rural electrification dataset.

    Args:
        output_path (str): Path where the generated CSV file will be saved.

    Returns:
        pd.DataFrame: Generated dataset with one row per village name.
    """
    # Set deterministic seeds so dataset generation stays reproducible.
    random.seed(42)
    np.random.seed(42)

    # Build village names and validate that the source list is not empty.
    village_names = _build_village_names()
    if not village_names:
        raise ValueError("Village name list must not be empty.")

    # Generate realistic population values in the requested range.
    populations = [random.randint(2000, 16000) for _ in village_names]

    # Generate realistic area values in square kilometers.
    areas = [round(random.uniform(10.0, 30.0), 2) for _ in village_names]

    # Compute population density from population and area.
    pop_density = [round(pop / area, 1) for pop, area in zip(populations, areas)]

    # Create distance values spanning 4.5 to 48 km, then shuffle to avoid sorting.
    distance_from_grid = np.linspace(4.5, 48.0, num=len(village_names)).round(2).tolist()
    random.shuffle(distance_from_grid)

    # Calculate electrification percentages with formula, noise, and clipping.
    electrification_pct = []
    for density, distance in zip(pop_density, distance_from_grid):
        score = 100 - (1.2 * distance) + (0.015 * density) + random.uniform(-5.0, 5.0)
        score = max(30.0, min(100.0, score))
        electrification_pct.append(round(score, 2))

    # Assemble the final DataFrame in the required column order.
    df = pd.DataFrame(
        {
            "village_name": village_names,
            "population": populations,
            "area_sqkm": areas,
            "pop_density": pop_density,
            "distance_from_grid_km": distance_from_grid,
            "electrification_pct": electrification_pct,
        }
    )

    # Create the destination directory if needed before saving.
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Save the generated dataset to CSV and print a confirmation line.
    df.to_csv(output_file, index=False)
    print(f"Dataset generated: {len(df)} rows saved to {output_file}")
    return df
