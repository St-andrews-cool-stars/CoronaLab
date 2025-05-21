
Create Synthetic Data Products from a 3D Stellar Corona Magnetic Field
----------------------------------------------------------------------

Developer Documentation
-----------------------

Installation
^^^^^^^^^^^^

From pip
    
.. code-block:: bash

    $ pip install corona_lab
    
From source
    
.. code-block:: bash

    $ git clone https://github.com/St-andrews-cool-stars/CoronaLab.git
    $ cd CoronaLab
    $ pip install .
    
For active developement intall in develop mode

.. code-block:: bash

    $ pip install -e .
    
Testing
^^^^^^^
Testing is run with `tox <https://tox.readthedocs.io>`_ (``pip install tox``).
Tests can be found in ``corona_lab/tests/``.

.. code-block:: bash

    $ tox -e test

Tests can also be run directly with pytest:

.. code-block:: bash

    $ pip install -e .[test]
    $ pytest

You can specify a single directory or file to test as:

.. code-block:: bash

    $ pytest corona_lab/tests/test_radio.py

Codestyle can be checked with:

.. code-block:: bash

    $ tox -e codestyle
    

Documentation
^^^^^^^^^^^^^
Documentation files are found in ``docs/``.

We build the documentation with `tox <https://tox.readthedocs.io>`_ (``pip install tox``):

.. code-block:: bash

    $ tox -e build-docs

You can also build the documentation with Sphinx directly using:

.. code-block:: bash
                
    $ pip install -e .[docs]
    $ cd docs
    $ make html
    
The built docs will be in ``docs/_build/html/``, to view them go to ``file:///path/to/corona_lab/repo/docs/_build/html/index.html`` in the browser of your choice.



License
-------

This project is Copyright (c) St Andrews Cool Stars Group and licensed under
the terms of the BSD 3-Clause license. This package is based upon
the `Openastronomy packaging guide <https://github.com/OpenAstronomy/packaging-guide>`_
which is licensed under the BSD 3-clause licence. See the `docs/LICENSE.rst` file.
more information.


Contributing
------------

We love contributions! CoronaLab is open source,
built on open source, and we'd love to have you hang out in our community.

**Imposter syndrome disclaimer**: We want your help. No, really.

There may be a little voice inside your head that is telling you that you're not
ready to be an open source contributor; that your skills aren't nearly good
enough to contribute. What could you possibly offer a project like this one?

We assure you - the little voice in your head is wrong. If you can write code at
all, you can contribute code to open source. Contributing to open source
projects is a fantastic way to advance one's coding skills. Writing perfect code
isn't the measure of a good developer (that would disqualify all of us!); it's
trying to create something, making mistakes, and learning from those
mistakes. That's how we all improve, and we are happy to help others learn.

Being an open source contributor doesn't just mean writing code, either. You can
help out by writing documentation, tests, or even giving feedback about the
project (and yes - that includes giving feedback about the contribution
process). Some of these contributions may be the most valuable to the project as
a whole, because you're coming to the project with fresh eyes, so you can see
the errors and assumptions that seasoned contributors have glossed over.

Note: This disclaimer was originally written by
`Adrienne Lowe <https://github.com/adriennefriend>`_ for a
`PyCon talk <https://www.youtube.com/watch?v=6Uj746j9Heo>`_, and was adapted by
model_corona based on its use in the README file for the
`MetPy project <https://github.com/Unidata/MetPy>`_.
