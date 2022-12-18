#!/usr/bin/python3

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))
# Build HTML make HTML
# Build Pdf make latex // pdflatex build/latex/structuraloptimizationgenerativedesign.tex
import os, sys
import shutil
current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,current_directory)
import conf_lib as conf_lib

on_rtd = os.environ.get('READTHEDOCS') == 'True'

clib = conf_lib.conf_lib()
data_dir = os.path.join(current_directory, "_data")
tables_dir = os.path.join(current_directory, "tables")
images_dir = os.path.join(current_directory, "images")
docs_dir = os.path.join(current_directory, "docs")

print("---------pymetamodels help build---------")
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
actual = os.path.dirname(os.path.abspath(__file__))
parent = os.path.abspath(os.path.join(actual, os.pardir))
grand_parent = os.path.abspath(os.path.join(parent, os.pardir))
sys.path.insert(0, parent)
sys.path.insert(0, grand_parent)

sub = os.path.abspath(os.path.join(os.path.join(parent,"src"), "pymetamodels"))
print("*-->*", sub)
sys.path.insert(0, os.path.abspath(sub))
sys.path.insert(0, os.path.abspath(os.path.join(sub, "clsplots")))

# -- Project information -----------------------------------------------------
project = r'pymetamodels package for materials, systems and component metamodeling'
copyright = '2021 ITAINNOVA - www.itainnova.es'
author = 'F Lahuerta'
show_authors = False
language = "en" #"es"

# The full version, including alpha/beta/rc tags
def setup(app):
    app.add_config_value('releaselevel', 'internal', 'external')

release = clib.theme_version(0,0,2)
releaselevel = "internal"

# source link
html_show_sourcelink = True
html_copy_source = True

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.ifconfig',
    'sphinx.ext.doctest',
    'sphinx_rtd_theme',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
    'sphinx.ext.autodoc',
    'myst_parser'
]

#extensions += ['rst2pdf.pdfbuilder']

# -- Autodoc
#autoclass_content = 'both'
autoclass_content = 'class'
autodoc_member_order = 'bysource'

# Bibliography
extensions += ['sphinxcontrib.bibtex']
bibtex_bibfiles = []
bibtex_default_style = 'unsrtalpha' #unsrt, plain, srt, unsrtalpha

bib_name_file = "001 DataScience.bib"
if on_rtd:
    pass
else:
    bib_folder = os.path.join(r"C:\Users\flahuerta\Documents\Mendely_Bibtex")
    bibtex_bibfiles += clib.add_bibligraphy_Mend(bib_name_file, bib_folder, current_directory)
    bib_folder = os.path.join(r"C:\Users\Francisco\Documents\Mendeley_Bibtex")
    bibtex_bibfiles += clib.add_bibligraphy_Mend(bib_name_file, bib_folder, current_directory)
    bib_folder = os.path.join(r"C:\Users\Paquito\Documents\Mendeley_Bibtex")
    bibtex_bibfiles += clib.add_bibligraphy_Mend(bib_name_file, bib_folder, current_directory)

# eq elements
numfig = True
math_number_all = True
math_numfig = True
imgmath_add_tooltips = True
math_eqref_format = "Eq.{number}"
numfig_secnum_depth = 4

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix of source filenames.
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Custom side bars
#html_sidebars = {'**': ['localtoc.html', 'navigation.html', 'relations.html', 'sourcelink.html', 'searchbox.html']}
# 'localtoc.html','sourcelink.html'
html_sidebars = {'**': ['about.html', 'navigation.html', 'relations.html', 'indexes.html', 'download.html','searchbox.html']}

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

if on_rtd:
    html_theme = 'default'
else:
    _theme = 3
    if _theme == 0:
        templates_path = [r"_templates",]
        html_theme_path = ["_templates",]
        html_theme = 'alabaster_ITA'
    elif _theme == 1:
        templates_path = [r"_templates",]
        html_theme_path = ["_templates\sphinx_rtd_theme_old", ]
        html_theme = "sphinx_rtd_theme"
    elif _theme == 2:
        html_theme = "sphinx_book_theme"
        html_theme_options = {
            "single_page": False,
            "home_page_in_toc": True,
            "show_navbar_depth": 10,
            "show_toc_level": 10,
        }
    elif _theme == 3:
        templates_path = [r"_templates",]
        html_theme_path = ["_templates", ]
        html_theme = "sphinx_rtd_theme_mod"
        html_sidebars = {'**': ['fulltoc.html', 'relations.html', 'sourcelink.html', 'searchbox.html']}
        html_theme_options = {
            'analytics_id': 'G-XXXXXXXXXX',  #  Provided by Google in your dashboard
            'analytics_anonymize_ip': False,
            'logo_only': False,
            'display_version': True,
            'prev_next_buttons_location': 'bottom',
            'style_external_links': False,
            'vcs_pageview_mode': '',
            'style_nav_header_background': '#55a5d9',
            # Toc options
            'collapse_navigation': False,
            'sticky_navigation': True,
            'navigation_depth': 10,
            'includehidden': True,
            'titles_only': False,
            'globaltoc_collapse': False,
            'globaltoc_maxdepth': 10,
        }
    elif _theme == 4:
        html_theme = "sphinx_rtd_theme"
        html_theme_options = {
            'analytics_id': 'G-XXXXXXXXXX',  #  Provided by Google in your dashboard
            'analytics_anonymize_ip': False,
            'logo_only': False,
            'display_version': True,
            'prev_next_buttons_location': 'bottom',
            'style_external_links': False,
            'vcs_pageview_mode': '',
            'style_nav_header_background': '#55a5d9',
            # Toc options
            'collapse_navigation': False,
            'sticky_navigation': True,
            'navigation_depth': 10,
            'includehidden': True,
            'titles_only': True,
        }

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_js_files = ['custom.js']
html_css_files = ['custom_js.css']

# Add variables to be used in the html context
html_context = {'alabaster_mod_version': clib.theme_version(0,0,1)}

# Create tables
file_name = "Tables"
path_xls = os.path.join(data_dir, file_name + ".xls")
sheet = "parse_tables_files"
clib.parse_tables_files(path_xls, sheet, tables_dir, data_dir)

# Create figures
file_name = "Images_A"
path_xls = os.path.join(data_dir, file_name + ".xls")
sheet = "parse_figures_files"
clib.parse_figures_files_table(path_xls, sheet, images_dir)
