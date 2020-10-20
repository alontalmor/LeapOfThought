
import pandas as pd
from LeapOfThought.common.file_utils import cached_path

class ConceptNet:
    """ A python singleton """

    class __impl:
        """ Implementation of the singleton interface """
        def __init__(self):
            pass

        def get_CENT_triplets(self):
            CENT_BERT_cached = cached_path('https://aigame.s3.us-west-2.amazonaws.com/data/CENT/CENT_BERT.csv.gz')
            return pd.read_csv(CENT_BERT_cached, compression='gzip')

        def get_ConceptNet_full(self):
            ConceptNet_full_cached = cached_path('https://aigame.s3.us-west-2.amazonaws.com/data/CENT/ConceptNet_edges.csv.gz')
            ConceptNet_full =  pd.read_csv(ConceptNet_full_cached, compression='gzip')
            # filter edges where start and end are the same
            ConceptNet_full = ConceptNet_full[ConceptNet_full['start_term'] != ConceptNet_full['end_term']]

            return ConceptNet_full

    # storage for the instance reference
    __instance = None

    def __init__(self):
        """ Create singleton instance """
        # Check whether we already have an instance
        if ConceptNet.__instance is None:
            # Create and remember instance
            ConceptNet.__instance = ConceptNet.__impl()

        # Store instance reference as the only member in the handle
        self.__dict__['_Singleton__instance'] = ConceptNet.__instance

    def __getattr__(self, attr):
        """ Delegate access to implementation """
        return getattr(self.__instance, attr)

    def __setattr__(self, attr, value):
        """ Delegate access to implementation """
        return setattr(self.__instance, attr, value)










