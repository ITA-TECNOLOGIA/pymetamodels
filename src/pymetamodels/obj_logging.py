#!/usr/bin/env python3

import os, logging


class objlogging(object):

    """Logging container

        :platform: Windows
        :synopsis: class related to the logging messages

        :Dependences: None

        |

    """

    def __init__(self, logging_path):

        if logging_path:
            self.logging_path =  os.path.join(logging_path,"logging_pymetamodels.log")
            logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filename=self.logging_path, encoding='utf-8', level=logging.INFO)


    def info(self, msg):

        """

        .. _logging_info:

        **Synopsis:**
            * Log with INFO level

        **Args:**
            * msg: message to log

        **Optional parameters:**
            * None

        **Returns:**
            * None

        .. note::

            * See tutorials :ref:`Tutorials <pymetamodels_tutoriales>`

        |

        """       

        logging.info(msg)

    def warning(self, msg):

        """

        .. _logging_warning:

        **Synopsis:**
            * Log with WARNING level

        **Args:**
            * msg: message to log

        **Optional parameters:**
            * None

        **Returns:**
            * None

        .. note::

            * See tutorials :ref:`Tutorials <pymetamodels_tutoriales>`

        |

        """       

        logging.warning(msg)        