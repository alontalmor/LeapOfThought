# pylint: disable=no-self-use,invalid-name
import pytest
import argparse
import os
from LeapOfThought.resources.teachai_kb import TeachAIKB
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class TestTeachAIKB:
    def test_gen_data(self):

        TeachAIKB().construct_kb()
