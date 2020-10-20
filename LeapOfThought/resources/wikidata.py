import pandas as pd
import requests
from LeapOfThought.common.file_utils import cached_path
from tqdm import tqdm
import pandas as pd

# This is mainly for testing and debugging  ...
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 2000)
pd.set_option('display.max_colwidth', 130)
pd.set_option("display.colheader_justify", "left")


class WikiData:
    """ A python singleton """

    class __impl:
        """ Implementation of the singleton interface """

        def __init__(self):
            self._sparql_url = 'https://query.wikidata.org/sparql'

        def run_query(self,query):
            r = requests.get(self._sparql_url, params={'format': 'json', 'query': query})
            if r.status_code != 200:
                return r.content
            else:
                data = r.json()
                query_data = []
                for item in data['results']['bindings']:
                    record = {}
                    for var in data['head']['vars']:
                        record[var.replace('Label', '')] = item[var]['value']
                    query_data.append(record)

                df = pd.DataFrame(query_data)
                df.drop_duplicates(inplace=True)
                ids_to_filter = []
                for col in df.columns:
                    ids_to_filter += list(df[df[col].str.contains('Q[0-9]+')].index)
                df = df[~df.index.isin(ids_to_filter)]
                return df

        def get_capitals(self, min_city_size):
            countries_query = """
            SELECT DISTINCT ?cityLabel ?countryLabel ?pop ?capitalLabel
            WHERE
            { 
              ?city wdt:P31/wdt:P279* wd:Q515 ; wdt:P1082 ?pop . FILTER (?pop > %d)
              ?city wdt:P17 ?country .
              ?country wdt:P36 ?capital.

              SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
            }
            LIMIT 100000
            """ % (min_city_size)
            print(countries_query)

            print('querying on sparql wikidata')
            r = requests.get(self._sparql_url, params={'format': 'json', 'query': countries_query})
            print(r)
            data = r.json()

            print('creating tabular data')
            countries_data = []
            for item in data['results']['bindings']:
                city = item['cityLabel']['value']
                country = item['countryLabel']['value']
                capital = item['capitalLabel']['value']
                pop = item['pop']['value']
                countries_data.append([city, country, capital, pop])

            df = pd.DataFrame(countries_data, columns=['city', 'country', 'capital','population'])
            df.drop_duplicates(inplace=True)
            return df

        def get_music_bands(self):
            countries_query = """
            SELECT DISTINCT ?bandLabel ?inception ?hasPartLabel WHERE {
              ?band wdt:P31 wd:Q215380 .
              ?band wdt:P136 ?genre .
              ?band wdt:P571 ?inception .
              ?band wdt:P527 ?hasPart .

              SERVICE wikibase:label {
                bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en" .
              }
            }
            """

            print('querying on sparql wikidata')
            r = requests.get(self._sparql_url, params={'format': 'json', 'query': countries_query})
            data = r.json()

            print('creating tabular data')
            countries_data = []
            for item in data['results']['bindings']:
                band = item['bandLabel']['value']
                inception = item['inception']['value']
                musician = item['hasPartLabel']['value']
                countries_data.append([band, inception, musician])

            df = pd.DataFrame(countries_data, columns=['band', 'inception', 'musician'])

            df['inception'] = pd.to_datetime(df['inception'], errors='coerce')

            df.drop_duplicates(inplace=True)
            return df

        def get_company_founders(self, min_employee_num):
            query =  """
                SELECT DISTINCT ?founderLabel ?companyLabel ?locationLabel ?employees
                WHERE
                {
                    ?company wdt:P112 ?founder ; wdt:P1128 ?employees . FILTER (?employees > %d)
                    ?founder wdt:P31 wd:Q5 .
                    ?company wdt:P159 ?location .
                    SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
                }
                LIMIT 10000
                """ % (min_employee_num)

            return self.run_query(query)

        def get_animals(self):
            query = """
            SELECT DISTINCT ?animalLabel ?mammalsub2Label ?mammalsub1Label
            WHERE
            {
                ?animal wdt:P279 ?mammalsub2 .
                ?mammalsub2 wdt:P279 ?mammalsub1 .
                ?mammalsub1 wdt:P279 wd:Q729 .
                SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
            }
            LIMIT 10000
            """

            query = """
            SELECT DISTINCT ?fishLabel 
            WHERE
            {
                ?fish wdt:P279 wd:Q3314483 .
                SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
            }
            LIMIT 10000
            """

            return self.run_query(query)

        def get_spouses(self):
            query = """
            SELECT DISTINCT ?personLabel ?spouseLabel
            WHERE
            {
                ?spouse wdt:P26 ?person .
                ?person wdt:P27 wd:Q30 .
                SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
            }
            LIMIT 10000
            """

            return self.run_query(query)

        def get_person_place_of_birth(self):
            query = """
            SELECT DISTINCT ?personLabel ?placeofbirthLabel
            WHERE
            {
                ?person wdt:P19 ?placeofbirth .
                SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
            }
            LIMIT 1
            """

            return self.run_query(query)

        def get_children(self):
            query = """
            SELECT DISTINCT ?personLabel ?childLabel ?countryLabel 
            WHERE
            {
              ?person wdt:P40 ?child .
              ?person wdt:P27 wd:Q30 .
              SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
            }
            LIMIT 20000
            """

            return self.run_query(query)

        def get_olmpics_composition_data(self):
            sparql_queries = [{'query': """
            SELECT DISTINCT ?bandLabel ?inception ?hasPartLabel WHERE {
              ?band wdt:P31 wd:Q215380 .
              ?band wdt:P136 ?genre .
              ?band wdt:P571 ?inception .
              ?band wdt:P527 ?hasPart .

              SERVICE wikibase:label {
                bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en" .
              }
            }
            """,
                               'object': 'hasPartLabel', 'inner': 'bandLabel', 'answer': 'inception',
                               'name': 'bands'},
                              {'query': """
            SELECT DISTINCT ?movieLabel ?personLabel ?spouseLabel WHERE {
               ?movie wdt:P31 wd:Q11424 .
               ?movie wdt:P161 ?person .
               ?person wdt:P26 ?spouse .

            #    ?narrative_location wdt:P625 ?coordinates .
              SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
            }
            LIMIT 10000
            """,
                               'object': 'movieLabel', 'inner': 'personLabel', 'answer': 'spouseLabel',
                               'name': 'movies'},
                              {'query': """
            SELECT DISTINCT ?personLabel ?companyLabel ?locationLabel
            WHERE
            {
                ?company wdt:P112 ?person .
                ?person wdt:P31 wd:Q5 .
                ?company wdt:P159 ?location .
                SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
            }
            LIMIT 10000
            """,
                               'object': 'personLabel', 'inner': 'companyLabel', 'answer': 'locationLabel',
                               'name': 'companies'}]

            all_data = []

            print('querying on sparql wikidata')
            for query in tqdm(sparql_queries):
                task_name = query['name']
                print('query: {}'.format(task_name))
                r = requests.get(self._sparql_url, params={'format': 'json', 'query': query['query']})
                print(r)
                data = r.json()

                print('creating tabular data')
                query_data = []
                for item in data['results']['bindings']:
                    inner = item[query['inner']]['value']
                    answer = item[query['answer']]['value']
                    obj = item[query['object']]['value']
                    query_data.append([inner, answer, obj, task_name])

                df = pd.DataFrame(query_data, columns=['inner', 'answer', 'object', 'name'])

                # particular handling dates
                if query['name'] == 'bands':
                    df['answer'] = pd.to_datetime(df['answer'], errors='coerce')
                    df['answer'] = df['answer'].dt.strftime('%Y')
                    df['answer'] = df['answer'].astype(str)

                df.drop_duplicates(inplace=True)
                df = df.drop(df[df.object.str.startswith('Q')].index)
                all_data.append(df)

            df = pd.concat(all_data)
            return df

    # storage for the instance reference
    __instance = None

    def __init__(self):
        """ Create singleton instance """
        # Check whether we already have an instance
        if WikiData.__instance is None:
            # Create and remember instance
            WikiData.__instance = WikiData.__impl()

        # Store instance reference as the only member in the handle
        self.__dict__['_Singleton__instance'] = WikiData.__instance

    def __getattr__(self, attr):
        """ Delegate access to implementation """
        return getattr(self.__instance, attr)

    def __setattr__(self, attr, value):
        """ Delegate access to implementation """
        return setattr(self.__instance, attr, value)










