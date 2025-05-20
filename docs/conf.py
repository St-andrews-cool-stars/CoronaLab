# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

import datetime
import os
import sys
import tomllib

with open(os.path.join(os.path.dirname(__file__), '..', 'pyproject.toml'), "rb") as FLE:
    conf = tomllib.load(FLE)

setup_cfg = conf['project']


# -- Project information -----------------------------------------------------

project = setup_cfg['name']
author = ','.join([x["name"] for x in setup_cfg['authors']])
copyright = '{0}, {1}'.format(
    datetime.datetime.now(datetime.timezone.utc).year, author)

__import__(project)
package = sys.modules[project]

# The short X.Y version.
version = package.__version__.split('-', 1)[0]
# The full version, including alpha/beta/rc tags.
release = package.__version__

#project = 'Model Corona'
#copyright = '2023, St Andrews Cool Stars Group'
#author = 'St Andrews Cool Stars Group'

# The full version, including alpha/beta/rc tags
#from model_corona import __version__
#release = __version__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.doctest',
    'sphinx.ext.mathjax',
    'sphinx_automodapi.automodapi',
    'sphinx_automodapi.smart_resolver',
    'sphinx_design'
]

# Add any paths that contain templates here, relative to this directory.
# templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# -- Options for intersphinx extension ---------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
#intersphinx_mapping = {'https://docs.python.org/': None}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'pydata_sphinx_theme'

html_theme_options = {
    "logo": {
        "image_light": "_static/banner.svg",
        "image_dark": "_static/banner.svg",
    },
    # https://github.com/pydata/pydata-sphinx-theme/issues/1492
    "navigation_with_keys": False,
    "navbar_align": "right"
}



# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".

html_static_path = ["_static"]
html_css_files = ["coronalab.css"]
html_favicon = "_static/icon.png"
html_copy_source = False

html_title = '{0} v{1}'.format(project, release)

# Output file base name for HTML help builder.
htmlhelp_basename = project + 'doc'

# By default, when rendering docstrings for classes, sphinx.ext.autodoc will 
# make docs with the class-level docstring and the class-method docstrings, 
# but not the __init__ docstring, which often contains the parameters to 
# class constructors across the scientific Python ecosystem. The option below
# will append the __init__ docstring to the class-level docstring when rendering
# the docs. For more options, see:
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#confval-autoclass_content
autoclass_content = "both"
