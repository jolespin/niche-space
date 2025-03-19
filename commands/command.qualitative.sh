job_name="qualitative"
/usr/bin/time -v python run_${job_name}.py 2> logs/${job_name}.e 1> logs/${job_name}.o
