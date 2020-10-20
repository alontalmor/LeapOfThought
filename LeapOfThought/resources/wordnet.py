
import pandas as pd
from LeapOfThought.common.file_utils import cached_path
from nltk.corpus import wordnet as wn
from tqdm import tqdm


class WordNet:
    """ A python singleton """

    class __impl:
        """ Implementation of the singleton interface """
        def __init__(self):
            pass

        def get_all_meanings(self, word):
            pass

        def get_all_wordnet_hypernyms(self, word, meaning_index_filter):
            output = []
            for syn in wn.synsets(word):
                original_syn = syn
                if original_syn.name().split('.')[-1] not in meaning_index_filter:
                    continue
                hypernyms = []
                while len(syn.hypernyms()) > 0:
                    hypernym = syn.hypernyms()[0]
                    syn = syn.hypernyms()[0]
                    if hypernym.name().split('.')[-1] not in meaning_index_filter:
                        continue
                    else:
                        hypernyms.append(hypernym)
                output.append((original_syn, hypernyms, word))
            return output

        def get_all_wordnet_hyponyms(self, word, meaning_index_filter, levels_down = 2, output_type = 'synsets'):
            output = []
            for syn in wn.synsets(word):
                original_syn = syn
                if original_syn.name().split('.')[-1] not in meaning_index_filter:
                    continue
                all_hyponyms = syn.hyponyms()
                queue = syn.hyponyms()
                for level in range(levels_down):
                    new_queue = []
                    for h in queue:
                        hyponyms = h.hyponyms()
                        all_hyponyms += [h for h in hyponyms if h.name().split('.')[-1] in meaning_index_filter]
                        new_queue += hyponyms
                    queue = new_queue
                if output_type == 'synsets':
                    output.append((original_syn, all_hyponyms, word))
                else:
                    output += [h.lemmas()[0].name() for h in all_hyponyms]
            return output

        def get_word_synonyms(self,word):
            synonyms = {}
            for i, syn in enumerate(wn.synsets(word)):
                for l in syn.lemmas():
                    if l.name() in synonyms:
                        synonyms[l.name()] += [syn]
                    else:
                        synonyms[l.name()] = [syn]
            return synonyms

        def get_synset_synonyms(self,syn):
            synonyms = {}
            for l in syn.lemmas():
                if l.name() in synonyms:
                    synonyms[l.name()] += [syn]
                else:
                    synonyms[l.name()] = [syn]

            return synonyms

        def get_word_antonyms(self,word):
            antonyms = {}
            for i, syn in enumerate(wn.synsets(word)):
                for l in syn.lemmas():
                    for antonym in l.antonyms():
                        if l.name() in antonyms:
                            antonyms[antonym.name()] += [syn]
                        else:
                            antonyms[antonym.name()] = [syn]
            return antonyms


        def get_all_antonyms_synonyms(self):
            synonyms = []
            antonyms = []
            for syn in tqdm(wn.all_synsets()):
                for i,l in enumerate(syn.lemmas()):
                    if i==0:
                        synonym_l1 = l
                    else:
                        synonyms.append({'word1': synonym_l1.name().replace('-', ' ').replace('_', ' '),
                                     'word2': l.name().replace('-', ' ').replace('_', ' '),
                                     'word1_pos': synonym_l1.synset().pos(),
                                     'word2_pos': l.synset().pos(),
                                     #'word1_lemma': synonym_l1,
                                     #'word2_lemma': l,
                                     'relation':'synonym'})

                    for antonym in l.antonyms():
                        antonyms.append({'word1':antonym.name().replace('-',' ').replace('_',' '),
                                         'word2':l.name().replace('-',' ').replace('_',' '),
                                         'word1_pos':antonym.synset().pos(),
                                         'word2_pos':l.synset().pos(),
                                         #'word1_lemma':antonym,
                                         #'word2_lemma': l,
                                         'relation':'antonym'})

            return antonyms, synonyms

    # storage for the instance reference
    __instance = None

    def __init__(self):
        """ Create singleton instance """
        # Check whether we already have an instance
        if WordNet.__instance is None:
            # Create and remember instance
            WordNet.__instance = WordNet.__impl()

        # Store instance reference as the only member in the handle
        self.__dict__['_Singleton__instance'] = WordNet.__instance

    def __getattr__(self, attr):
        """ Delegate access to implementation """
        return getattr(self.__instance, attr)

    def __setattr__(self, attr, value):
        """ Delegate access to implementation """
        return setattr(self.__instance, attr, value)










