# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 16:16:01 2020

@author: dzygmunt
"""

#pandas - dtruktura dataframe
import pandas as pd
#re - regular expressions
import re
#sys - sys.argv, sys.modules
import sys
#spacy - trzon analizy jezykowej
import spacy as sp

def Count_words_freq(text, nlp_pl):
    #potrzeba: morfeusz2, pl_spacy_model_morfeusz_big, tensorflow
    #df - tabela wszystkich slow wraz z ich cechami
    #po kolei caly dokument
    MyDocument=nlp_pl(text)
    attribs = ['orth_', 'lemma_', 'tag_', 'pos_', 'dep_', 'head']
    table = [{att:tok.__getattribute__(att) for att in attribs} for tok in MyDocument]
    df = pd.DataFrame(table)
    del(table, attribs)
    
    #df2 - okrojona wersja df, bez pewnych slow-smieci
    all_rows=list(range(0,len(df)))
    df2=df.drop(all_rows)
    
    for word in all_rows:
        #df.iloc[:, 0] <=> df.orth_ <=> oryginalne slowo z tekstu
        #df.iloc[:, 1] <=> df.lemma_ <=> slowo w bazowej formie
        #df.iloc[:, 3] <=> df.pos_ <=> czesc mowy
        #temporary_logical - indeks slow zakazanych:
        #stopword lub znak interpunkcyjny lub wykrzyknik lub 1+ "\n" lub
        #1+ cyfry wraz z 0+ kropka na koncu lub "np."
        temporary_logical=MyDocument.vocab[df.iloc[word, 0]].is_stop or df.iloc[word, 3]=="PUNCT"\
            or df.iloc[word, 3]=="INTJ" or (re.search("^[\n]+$", df.iloc[word, 1]) is not None)\
            or (re.search("^[0-9]+[\.]*$", df.iloc[word, 1]) is not None)\
            or (re.search("^np\.$", df.iloc[word, 1]) is not None)
        if not temporary_logical:
            new_row=df.loc[word]
            df2=df2.append(new_row, ignore_index=True)
            del(new_row)
        del(temporary_logical)
    del(all_rows, word, df)
    
    base_words=list(df2.iloc[:, 1]) 
    all_rows=list(range(0, len(base_words)))
    word_frequency={}
    
    #przegladanie slow jako listy, jesli slowa nie ma w slowniku
    #word_frequency, to dodaj je jako indeks, a pod nim lista
    #jesli jest w slowniku (jako indeks), to zwieksz licznik
    for word_number in all_rows:
        word_text=base_words[word_number]
        if word_text not in word_frequency.keys():
            #czyli {}[word]=[part of speech, licznik, liczba slow w dokumencie, -||-]
            word_frequency[word_text]=[df2.iloc[word_number, 3], 1, len(MyDocument), len(MyDocument)]
        else:
            word_frequency[word_text][1]+=1
        del word_text    
    del(base_words, word_number, df2, MyDocument)
    
    #opisowe nazwy kolumn w df na bazie slownika
    df_words_freq=pd.DataFrame.from_dict(word_frequency, orient="index", columns=["Part of speech", "Counts", "Frequency", "Document's length"])
    all_rows=list(range(0, len(df_words_freq)))
    
    #techniczne - oblicza czestotliwosc=zliczenia/dlugosc dokumentu + zaokraglenie
    for number in all_rows:
        df_words_freq.iloc[number, 2]=\
            round(df_words_freq.iloc[number, 1]/df_words_freq.iloc[number, 3], 3)
    del(number, all_rows, word_frequency)
    #sortowanie po liczbie zliczen
    df_words_freq=df_words_freq.sort_values(by="Counts", ascending=False)
    df_words_freq=df_words_freq.assign(Position=list(range(1,(1+len(df_words_freq)))))
    #print(df_words_freq)
    return df_words_freq

def Count_phrases_freq(text, nlp_pl):
    #potrzeba: morfeusz2, pl_spacy_model_morfeusz_big, tensorflow
    #po kolei caly dokument
    MyDocument=nlp_pl(text)
    #word.i <=> numer slowa w dokumencie
    #word.tag_ <=> "czesc zdania" - subst to zazwyczaj podmiot lub dopelnienie
    #word.head/children <=> rodzic lub dziecko w tej strukturze drzewa (rozbior logiczny zdania)
    #lista subst jako numerow slow w zdaniu
    #if: jestes subst i nie jestes jedynym dzieckiem susbt-a
    list_subst=[word.i for word in MyDocument\
        if (word.tag_=="subst" and (not \
        (word.head.tag_=="subst" and len(list(word.head.children))<2)))]
    
    #list_sets <=> lista list, np:
    #[[[0, 1, 2], 1], [[3, 4, 5], 3], [[6, 7], 7]]
    #lista dwuelementowych list: poddrzew subst-ow + subts.i
    #kazde poddrzewo jest niewlasciwe w sensie zawierania (subst do niego nalezy)
    #poddrzewo jest reprezentowane przez liste - numery slow
    list_sets=[[[ _.i for _ in MyDocument[subst].subtree], subst] for subst in list_subst]
    temp_setlist=list_sets.copy()
    del list_subst
    
    for set1 in list_sets:
        for set2 in list_sets:
            #set1/2[0] - poddrzewo
            #if set2[0] jest podzbiorem wlasciwym set1[0]: 
            if set(set1[0])>set(set2[0]):
                #roznica symetryczna - w tym przypadku
                #nadmiar set1 pod wyjebaniu set2
                temporary=list(set(set1[0]).symmetric_difference(set(set2[0])))
                #kazde drzewo ma numerki (slow), są one sortowane
                #w kolejnosci rosnacej, czyli jak w zdaniu/dokumencie
                temporary.sort()
                temp_setlist.append([temporary, set1[1]])
                del temporary
    #list_sets <=> teraz ma wszystkie drzewa, a takze ich symetryczne roznice
    list_sets=temp_setlist
    del set1, set2, temp_setlist
    
    #word.lemma_ <=> bazowa forma slowa
    #word.text <=> forma string-a, male litery 
    #list_results: lista 2-elementowych list
    #string - slowa bazowe w alfabetycznej kolejnosci | string - oryginal z tekstu
    list_results=[]
    for set1 in list_sets:
        #if: set1[0] ma 2 do 5 slow
        if len(set1[0])>1 and len(set1[0])<6:
            #if: jesli rodzicem wierzcholka drzewa jest subst
            #i nie jest on sam soba (ROOT jest rodzicem sam dla siebie w tej konwencji)
            #to mozliwe, bo ROOT to czasownik/orzeczenie w zdaniu,
            #ale w rownowaznikach zdan jego trzonem najczesciej jest subst
            if (MyDocument[set1[1]].head.tag_=="subst") and\
                (not (set1[1]==MyDocument[set1[1]].head.i)):
                set1[0].append(MyDocument[set1[1]].head.i)
            set1[0].sort()
            #zamieniam numery slow na slowa-tekst, zachowujac kolejnosc
            #opuszczam pierwsze i ostatnie slowo gdy jest interpunkcją lub "!"
            temp_phrases=[MyDocument[_].text for counter, _ in enumerate(set1[0])\
                if not ((counter==0 or counter==(len(set1[0])-1)) \
                and (MyDocument[_].pos_=="PUNCT" or MyDocument[_].pos_=="INTJ"))]
            temp_lemma=[MyDocument[_].lemma_ for counter, _ in enumerate(set1[0])\
                if not ((counter==0 or counter==(len(set1[0])-1)) \
                and (MyDocument[_].pos_=="PUNCT" or MyDocument[_].pos_=="INTJ"))]
            #liste z bazowymi formami sortujemy domyslnie
            #alfabetycznie i rosnaco
            temp_lemma.sort()
            #zlepiamy liste stringow w jeden string, z separatorem: spacja=' '
            temp_phrases=' '.join(temp_phrases)
            temp_lemma=' '.join(temp_lemma)
            list_results.append([temp_lemma, temp_phrases])
            del temp_phrases, temp_lemma
    del set1, list_sets
    
    #lista bazowych form - jako stringow
    base_phrases=[_[0] for _ in list_results]
    phrases_frequency={}
    
    #przegladanie fraz jako listy, jesli frazy nie ma w slowniku
    #phrases_frequency, to dodaj fraze (bazowa) jako indeks, a pod nia liste
    #jesli juz jest w slowniku, to zwieksz indeks
    for counter, phrase in enumerate(base_phrases):
        if phrase not in phrases_frequency.keys():
            #czyli {}[phrase]=[forma z tektu, licznik, liczba slow w dokumencie, -||-]
            phrases_frequency[phrase]=[list_results[counter][1], 1, len(MyDocument), len(MyDocument)]
        else:
            phrases_frequency[phrase][1]+=1
    del counter, phrase, base_phrases, list_results, MyDocument
    
    #opisowe nazwy kolumn w df na bazie slownika
    df_phrases_freq=pd.DataFrame.from_dict(phrases_frequency, orient="index", columns=["Phrase in text", "Counts", "Frequency", "Document's length"])
    all_rows=list(range(0, len(df_phrases_freq)))
    
    #techniczne - oblicza czestotliwosc=zliczenia/dlugosc dokumentu + zaokraglenie
    for number in all_rows:
        df_phrases_freq.iloc[number, 2]=\
            round(df_phrases_freq.iloc[number, 1]/df_phrases_freq.iloc[number, 3], 3)
    del number, all_rows, phrases_frequency
    #sortowanie po liczbie zliczen
    df_phrases_freq=df_phrases_freq.sort_values(by="Counts", ascending=False)
    df_phrases_freq=df_phrases_freq.assign(Position=list(range(1,(1+len(df_phrases_freq)))))
    
    return df_phrases_freq
    
def Calculate_positions_df(list_of_df_WF, list_of_md5, how_many_documents):
    #lista list - dla kazdego dokumentu lista
    #lista slow/fraz - z indeksow df
    temp_list=[]
    for sth in list_of_df_WF:
        temp_list.append(list(sth.index))
    del(sth)
    #przeciecie list slow/fraz z wszystkich dokumentow
    intersection_set=list(set(temp_list[0]).intersection(*temp_list[0:how_many_documents]))        
    del temp_list
    #slownik z listami
    #indeks - bazowe slowo lub fraza
    calculated_positions={}
    #if: przeciecie jest niepuste
    if len(intersection_set)>0:
        for word in intersection_set:
            #temp_count -> global counts
            #temp_density -> global density
            #temp_position -> average position
            temp_count=0
            temp_density=0
            temp_position=0
            #struktura dictionary
            #word -> part of speech/ average position/ glocal counts/ global density
            #phrase -> pierwsza przyklad frazy z tekstu/ -||-
            calculated_positions[word]=[list_of_df_WF[0].loc[word][0],\
                                        temp_position, temp_count, temp_density]
            #dla dokumentow po kolei
            for doc_num in list(range(0, how_many_documents)):
                #word -> -||-/ document md5/ local counts/ local density
                calculated_positions[word].append(list_of_md5[doc_num])
                calculated_positions[word].append(list_of_df_WF[doc_num].loc[word][1])
                calculated_positions[word].append(list_of_df_WF[doc_num].loc[word][2])
                #dodaje do globalnych wartosci
                temp_count+=list_of_df_WF[doc_num].loc[word][1]
                temp_density+=list_of_df_WF[doc_num].loc[word][3]
                temp_position+=list_of_df_WF[doc_num].loc[word][4]
                
            #wylicza globalne wartosci
            temp_density=round(temp_count/temp_density, 4)
            temp_position=round(temp_position/how_many_documents, 4)
            #i je wpisuje
            calculated_positions[word][1]=temp_position
            calculated_positions[word][2]=temp_count
            calculated_positions[word][3]=temp_density
            del(doc_num, temp_count, temp_density, temp_position)
        del word
    del intersection_set
    
    #if: slownik niepusty
    if len(calculated_positions)>0:
        #utworz df prawdziwy
        df_MyWords=pd.DataFrame.from_dict(calculated_positions, orient="index")
    else:
        #utworz df pusty o 4 kolumnach
        df_MyWords=pd.DataFrame.from_dict(calculated_positions, orient="index", columns=range(0, 4))
    #techniczne: zmienia nazwy kolumn w stringi - by móc wpisac ladne nazwy
    df_MyWords.columns = df_MyWords.columns.map(str)
    
    del calculated_positions
    
    return df_MyWords

#wczytanie skryptu z parametrami
#sys.argv[0] <=> nazwa skryptu, istnieje zawsze!
arguments=sys.argv

#tworzenie dataframe - df - z potrzebnym info
if len(arguments)>1:
    df_DB2=pd.read_csv(arguments[1], names=["md5", "text"])
else:
    import mysql.connector as sqlcon
    DB_password=""
    rows_number=1000
    myDB=sqlcon.connect(user='dzygmunt', password=DB_password, database='dawid', host='mysql.senuto.com', port=3306)
    cursor=myDB.cursor()
    question=("""SELECT rba.body content, rba.length how_long, ra.url_finish links, rba.uid
    FROM crawler_urls_checker.results_body_algo rba
    JOIN crawler_urls_checker.results_algo ra
    ON rba.uid=ra.uid
    WHERE rba.length>1000
    ORDER BY rba.uid
    LIMIT %s""")
    
    cursor.execute(question, params=(rows_number, ))
    
    df_DB=pd.DataFrame(cursor)
    
    cursor.close()
    myDB.close()
    del(myDB, rows_number, question, cursor, DB_password)
    
    all_rows=list(range(0, len(df_DB)))
    df_DB2=df_DB.drop(all_rows)
    
    #usuwanie niektorych tekstow przy recznym wczytaniu z bazy danych
    for r_num in all_rows:
        #w stringu url-a istnieje ".pl" and
        # nie ma znaku "�" - efekt zjebanego kodowania
        temporary_logical=(re.search("\.pl\/", df_DB[2][r_num]) is not None)\
            and (re.search("�", df_DB[0][r_num]) is None)
        if temporary_logical:
            new_row=df_DB.loc[r_num]
            df_DB2=df_DB2.append(new_row, ignore_index=True)
            del(new_row)
        del(temporary_logical)
    del(r_num, all_rows, df_DB)

#wczytanie modulu z analiza jezyka polskiego    
if (not ("pl_spacy_model_morfeusz_big" in sys.modules)):
    nlp_pl=sp.load("pl_spacy_model_morfeusz_big")

#listy df dla dokumentow - w inputowej kolejnosci
#WF - words frequency
#PF - phrases frequency    
list_of_df_WF=[]
list_of_df_PF=[]
i=0
while (i<len(df_DB2)):
    if len(arguments)>1:
        #df_DB2.iloc[i, 1] - tekst przy wczytaniu z pliku
        df_WordFreq=Count_words_freq(df_DB2.iloc[i, 1], nlp_pl)
        df_PhraseFreq=Count_phrases_freq(df_DB2.iloc[i, 1], nlp_pl)
    else:
        #df_DB2.iloc[i, 0] - tekst przy wczytaniu z bazy
        df_WordFreq=Count_words_freq(df_DB2.iloc[i, 0], nlp_pl)
        df_PhraseFreq=Count_phrases_freq(df_DB2.iloc[i, 0], nlp_pl)
    #dzieki wzrastaniu i mamy zachowana kolejnosc dokumentow
    list_of_df_WF.append(df_WordFreq)
    list_of_df_PF.append(df_PhraseFreq)
    i+=1
    del(df_WordFreq, df_PhraseFreq)
del i

#list_of_md5 - lista md5, ktore indentyfikuja dokument
#kolejnosci dokumentow zgodna z ta w powyzszym while, ze wzgledu na konstrukcje
if len(arguments)>1:
    list_of_md5=[_ for _ in df_DB2.iloc[:, 0]]
else:
    list_of_md5=[_ for _ in df_DB2.iloc[:, 3]]
#X - z ilu dokumentow robic przekroj?
#przekroj <=> intersection
#X=2
X=len(df_DB2)
#jeden df z words/phrases ze wszystkich dokumentow
df_MyWords=Calculate_positions_df(list_of_df_WF, list_of_md5, X)
df_MyPhrases=Calculate_positions_df(list_of_df_PF, list_of_md5, X)
del list_of_df_WF, list_of_df_PF, list_of_md5, X

#ladne nazwy 4 pierwszych kolumn:
names_w=["Part of speech", "Average position", "Global counts", "Global density"]
names_p=["Phrase example", "Average position", "Global counts", "Global density"]
#struktura tych df:
#indeks - word/phrase w bazowej formie/ 4 powyzsze kolumny/
#po 3 kolumny na kazdy dokument: md5 dokumentu/ local counts/ local density
i=0
while i<len(names_w):
    df_MyWords.columns.values[i]=names_w[i]
    df_MyPhrases.columns.values[i]=names_p[i]
    i+=1
del i

#sortowanie wg "Average position"
df_MyWords=df_MyWords.sort_values(by=names_w[1], ascending=True)
df_MyPhrases=df_MyPhrases.sort_values(by=names_p[1], ascending=True)

#wpisywanie do pliku
if len(arguments)>2:
    df_MyWords.to_csv(arguments[2])
if len(arguments)>3:
    df_MyPhrases.to_csv(arguments[3])

del names_w, names_p


