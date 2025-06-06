"""
General script for active learning with GPR.
"""
import os 
from ipi.utils.scripting import (
    InteractiveSimulation
)
from ipi.utils.depend import dstrip 
import ase, ase.io 
from optparse import OptionParser
import gpr_interface


def parse_input():
    """
    copy from /bin/i-pi.
    """
    parser = OptionParser(usage='%prog [options] <input file>',
                          description='The main i-PI executable used to run '
                                      'a simulation, given an XML input file.'
                          )
    
    parser.add_option('-V', '--verbosity', dest='verbosity', default=None,
                    choices=['quiet', 'low', 'medium', 'high', 'debug'],
                    help='Define the verbosity level.')
    
    options, args = parser.parse_args()

    # make sure that we have exactly one input file and it exists
    if len(args) == 0:
        parser.error('No input file name provided.')
    elif len(args) > 1:
        parser.error('Provide only one input file name.')
    else:
        fn_in = args[0]
        if not os.path.exists(fn_in):
            parser.error('Input file not found: {:s}'.format(fn_in))

    return fn_in, options 

def main(sim: InteractiveSimulation):
    """
    active learning for path searching.
    """
    motion = sim.syslist[0].motion 
    
    active_motion = gpr_interface.ActiveLearning(sim = sim,
                                                 motion= motion)
    # initialize gpr model.
    active_motion.initialize_gpr_model()

    # run the i-pi simulation and update the gpr model.
    active_motion.run()

if __name__ == '__main__':
    # parse input 
    fn_in, options = parse_input() 
    # initialize simulation object.
    sim = InteractiveSimulation(fn_in)

    main(sim)
    