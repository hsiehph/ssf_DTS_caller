ALL: 
	. ./modules.sh; python setup.py build_ext --inplace --include-dirs ./heap 
	
clean:
	rm ./c_hierarchical_edge_merge.cpp ./traverse_contours.c ./c_hierarchical_edge_merge.so ./traverse_contours.so get_windowed_variance.c get_windowed_variance.so 
	
