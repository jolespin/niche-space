#for organism_type in "prokaryotic" "eukaryotic"
for organism_type in "eukaryotic"
do
	job_name="skani__${organism_type}"
	input_filepath="../data/cluster/ani/genome_filepaths.${organism_type}.list"
	output_filepath="../data/cluster/ani/skani_output.${organism_type}.tsv"
	/usr/bin/time -v skani triangle --sparse -t 1 -l ${input_filepath} -o ${output_filepath} --ci --min-af 15.0 -s 80.0 -c 125 -m 1000 --medium 2> logs/${job_name}.e 1> logs/${job_name}.o
done
