sector_codes = {
    # Codes from the first image (the "continued" part)
    "TRDret": "Retail trade services, except of motor vehicles and motorcycles", # Corresponds to NACE G47
    "TRA+": "Transportation and storage",
    "TRAinl": "Land transport and transport via pipelines",
    "TRAwat": "Water transport",
    "TRAair": "Air transport",
    "TRAwar": "Warehousing and support activities for transportation",
    "TRApos": "Postal and courier activities",
    "FD+": "Accommodation and food service activities",
    "COM+": "Information and communication",
    "COMpub": "Publishing activities",
    "COMvid": "Motion picture, video and television production, sound recording, broadcast-ing",
    "COMtel": "Telecommunications",
    "COMcom": "Computer programming, consultancy; Information service activities",
    "FIN+": "Financial and insurance activities",
    "FINser": "Financial services, except insurance and pension funding",
    "FINins": "Insurance, reinsurance and pension funding services, except compulsory social security",
    "FINaux": "Activities auxiliary to financial services and insurance services",
    "RES+": "Real estate activities",
    "PRO+": "Professional, scientific and technical activities",
    "PROleg": "Legal and accounting services; Activities of head offices; management consultancy activities",
    "PROeng": "Architectural and engineering activities; technical testing and analysis",
    "PROsci": "Scientific research and development",
    "PROadv": "Advertising and market research",
    "PROoth": "Other professional, scientific and technical activities; Veterinary activities",
    "ADM+": "Administrative and support service activities",
    "PUB+": "Public administration and defence; compulsory social security",
    "EDU+": "Education",
    "HEA+": "Human health and social work activities",
    "ART+": "Arts, entertainment and recreation",
    "HOU+": "Activities of households as employers"
}

# Codes from the new image (the start of the NACE level 2 sectors table)
new_sector_codes = {
    "AGR+": "Agriculture, forestry and fishing",
    "AGRagr": "Crop and animal production, hunting and related service activities",
    "AGRfor": "Forestry and logging",
    "AGRfis": "Fishing and aquaculture",
    "MIN+": "Mining and quarrying",
    "MINfos": "Mining and extraction of energy producing products",
    "MINoth": "Mining and quarrying of non-energy producing products",
    "MINsup": "Mining support service activities",
    "MAN+": "Manufacturing",
    "MANfoo": "Food, beverages and tobacco products",
    "MANtex": "Textiles, wearing apparel, leather and related products",
    "MANwoo": "Wood and products of wood and cork, except furniture",
    "MANpap": "Paper and paper products",
    "MANpri": "Printing and reproduction of recorded media",
    "MANref": "Coke and refined petroleum products",
    "MANche": "Chemicals and chemical products",
    "MANpha": "Basic pharmaceutical products and pharmaceutical preparations",
    "MANpla": "Rubber and plastic products",
    "MANmin": "Other non-metallic mineral products",
    "MANmet": "Basic metals",
    "MANfmp": "Fabricated metal products, except machinery and equipment",
    "MANcom": "Computer, electronic and optical products",
    "MANele": "Electrical equipment",
    "MANmac": "Machinery and equipment n.e.c.",
    "MANmot": "Motor vehicles, trailers and semi-trailers",
    "MANtra": "Other transport equipment",
    "MANfur": "Furniture and other manufactured goods",
    "MANrep": "Repair and installation services of machinery and equipment",
    "PWR+": "Electricity, gas, steam and air conditioning",
    "WAT+": "Water supply; sewerage; waste management and remediation",
    "WATwat": "Natural water; water treatment and supply services",
    "WATwst": "Sewerage services; sewage sludge; waste collection, treatment and disposal services",
    "CNS+": "Constructions and construction works",
    "TRD+": "Wholesale and retail trade; repair of motor vehicles and motorcycles", # NACE G
    "TRDmot": "Wholesale and retail trade and repair services of motor vehicles and motorcycles", # NACE G45
    "TRDwho": "Wholesale trade, except of motor vehicles and motorcycles" # NACE G46
}

# Update the original dictionary with the new codes
# If there are any overlapping keys, the values from new_sector_codes will take precedence.
sector_codes.update(new_sector_codes)

check_list = [
    "ADM+", "AGRagr", "AGRfis", "AGRfor", "ART+", "CNS+", "COMcom", "COMpub",
    "COMvid", "EDU+", "ENRcoa", "ENRele", "ENRgas", "ENRoil", "EXT+", "FD+",
    "FINser", "HEA+", "MANche", "MANcom", "MANele", "MANfmp", "MANfoo", "MANfur",
    "MANmac", "MANmet", "MANmin", "MANmot", "MANpap", "MANpha", "MANpla", "MANpri",
    "MANrep", "MANtex", "MANtra", "MANwoo", "MIN+", "PROleg", "PUB+", "RES+",
    "TRAair", "TRAinl", "TRApos", "TRAwar", "TRAwat", "TRDmot", "WATwat", "WATwst"
]

# Convert the list and dictionary keys to sets for efficient comparison
check_set = set(check_list)
dictionary_keys_set = set(sector_codes.keys())

# Codes in check_list but not in dictionary
not_in_dictionary = check_set - dictionary_keys_set
print("Codes from the list NOT present in the dictionary:")
if not_in_dictionary:
    for code in sorted(list(not_in_dictionary)): # Sort for consistent output
        print(code)
else:
    print("None")

print("\n--------------------------------------------------\n")

# Codes in dictionary but not in check_list
not_in_check_list = dictionary_keys_set - check_set
print("Codes present in the dictionary but NOT in the list:")
if not_in_check_list:
    for code in sorted(list(not_in_check_list)): # Sort for consistent output
        print(code)
else:
    print("None")

print("\n--- Summary ---")
print(f"Total codes in the list: {len(check_list)}")
print(f"Total unique codes in the list: {len(check_set)}")
print(f"Total codes in the dictionary: {len(sector_codes)}")
print(f"Codes from list found in dictionary: {len(check_set.intersection(dictionary_keys_set))}")
print(f"Codes from list NOT found in dictionary: {len(not_in_dictionary)}")
print(f"Codes from dictionary NOT found in list: {len(not_in_check_list)}")