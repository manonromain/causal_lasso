sphinx-apidoc -f -o source ../graph_tools/
mv source/modules.rst source/modules_graph_tools.rst
sphinx-apidoc -f -o source ../causal_lasso/
make html
