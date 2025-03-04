quality_label="completeness_gte50.contamination_lt10"
for organism_type in "prokaryotic" "eukaryotic"
do
    # Set the thread count based on the organism type
    if [ "$organism_type" = "prokaryotic" ]; then
        threads=48
    else
        threads=1
    fi

    job_name="skani__${organism_type}"
    input_filepath="../data/cluster/ani/v2025.3.3/${organism_type}/${quality_label}/genome_filepaths.list"
    output_filepath="../data/cluster/ani/v2025.3.3/${organism_type}/${quality_label}/skani_output.tsv"
    echo "${organism_type} ${threads}"
    /usr/bin/time -v skani triangle --sparse -t ${threads} \
        -l ${input_filepath} -o ${output_filepath} \
        --ci --min-af 15.0 -s 80.0 -c 125 -m 1000 --medium \
        2> logs/${job_name}.e 1> logs/${job_name}.o
done
