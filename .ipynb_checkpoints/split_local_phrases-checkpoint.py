import pandas as pd
import spacy as sp
# import re
# import sys
# from spacy.lang.pl.stop_words import STOP_WORDS


# stara, nieefektywna funkcja

# def clean_list(l1, l2, protected = ""):
#     result = []
#     for i in l1:
#         counter = 0
#         for j in l2:
#             ############################################################################################
#             ###  zwiększamy licznik jeśli element z 2 listy nie znajduje się w elemencie z 1 listy   ###
#             ###  lub jeśeli element z 2 listy jest chorniony ###########################################
#             ############################################################################################
#             print(f'Element z 1 listy: "{i}", element z 2 listy: "{j}", a licznik to: {counter}')
#             print((j.lower() not in i.lower()), (j.lower() in protected.lower()))
#             print("\n\n")
#             if (j.lower() not in i.lower()) or (j.lower() in protected.lower()):
#                 counter += 1
#             print(len(l2), f"Licznik to: {counter}")
#             if counter == len(l2):
#                 print(f'Wpisuję frazę {i} do listy wynikowej')
#                 result.append(i)
#     return result


# Wczytywanie listy z pliku

def load_file_to_list(filename):
    result = []
    with open(filename, encoding='utf-8') as fh:
        for line in fh:
            clean_line = line.replace("\n", "")
            result.append(f'{clean_line}')
    return result


# zapisywanie listy do pliku

def write_list_to_file(input_list, filename):
    with open(filename, 'w', encoding='utf-8') as fh:
        for element in input_list:
            fh.write(f"{element}\n")

            
# funkcja zapisuje do pliku excel 3 arkusze: pierwotny, frazy bez miast oraz frazy z miastami
# przyjmuje 3 parametry
# dataframe
# tablicę boolowską pozwalającą rozrżnić frazy
# ścieżkę do zapisu pliku

def write_to_excel(df, bool_table, filepath):
    with pd.ExcelWriter(filepath) as writer:
        df.to_excel(writer, sheet_name='Wszystkie frazy')
        df[bool_table].to_excel(writer, sheet_name='Frazy bez miast')
        df[~bool_table].to_excel(writer, sheet_name='Frazy z miastami')
            

# funkcja, która pozwoli później stworzyć tablicę boolowską
# na podstawie wcześniej wczytanej listy z miastami
# False - gdzie jest miasto, True - gdzie nie ma miasta

def filter_citys(row):
    citys = load_file_to_list("Input/lista_miast.txt")
    for city in citys:
        if city.lower() in row["Keyword"].lower():
            return False
    return True


# wczytanie pliku csv jako dataframe
df = pd.read_csv("Input/all-keyword-ideas2.csv", engine='c')


# tworzenie tablicy boolowskiej za pomocą wbudowanej funkcji pandas - apply
# axis=1 oznacza, że funkcja będzie wykonywana na wierszach, a nie na kolumnach
bool_t = df.apply(filter_citys, axis=1)


# wywołanie funkcji zapisu
write_to_excel(df, bool_t, "Output/podzielona_lista.xlsx")
