for organism_type in "prokaryotic" "eukaryotic"
do 
    job_name="cluster-ani__${organism_type}"
    input_filepath="../data/cluster/ani/${organism_type}/skani_output.tsv"
    output_directory="../data/cluster/ani/${organism_type}/completeness_gte90.contamination_lt5"
    identifiers_filepath="../data/cluster/ani/${organism_type}/completeness_gte90.contamination_lt5/organisms.list"
    graph_filepath="../data/cluster/ani/${organism_type}/completeness_gte90.contamination_lt5/networkx_graph.pkl.gz"
    dict_filepath="../data/cluster/ani/${organism_type}/completeness_gte90.contamination_lt5/dict.pkl.gz"
    representatives_filepath="../data/cluster/ani/${organism_type}/completeness_gte90.contamination_lt5/representatives.tsv.gz"

    mkdir -p "${output_directory}"

    # Grab the first character of organism_type, make it uppercase, and build the cluster_prefix.
    prefix=$(echo "${organism_type}" | cut -c1 | tr '[:lower:]' '[:upper:]')
    cluster_prefix="NAL-${prefix}SLC_"

    # Build parameters string with the dynamically generated cluster_prefix.
    params="--basename \
	-t 95 \
	-a 50 \
	--af_mode relaxed \
	--cluster_prefix ${cluster_prefix} \
	-o ${output_directory}/genome_clusters.tsv \
	--identifiers ${identifiers_filepath} \
	--export_graph ${graph_filepath} \
	--export_dict ${dict_filepath} \
	--export_representatives ${representatives_filepath}"

    # Run the command with the constructed parameters.
    cat "${input_filepath}" | cut -f1-5 | tail -n +2 | python ../bin/edgelist_to_clusters.py ${params} 2> logs/${job_name}.e 1> logs/${job_name}.o

done

