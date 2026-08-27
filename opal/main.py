# SPDX-License-Identifier: Apache-2.0
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from opal.utils.util import get_bool_env_var
from opal.utils.opal_profile import profile_function
from opal.core.opal import OpalSimulator

if __name__ == "__main__":
    print(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    print(sys.path)
    opal = OpalSimulator.from_cmd_args()

    if True == get_bool_env_var("OPAL_PROFILE", False):
        runtime, virtual_time = profile_function(opal.run)
    else:
        runtime, virtual_time = opal.run()

    del opal  # cleanup
