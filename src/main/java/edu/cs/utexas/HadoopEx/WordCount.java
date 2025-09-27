package edu.cs.utexas.HadoopEx;

import java.io.IOException;
import java.util.Locale;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

/**
 * Entry point that wires together the MapReduce jobs for the three assignment tasks.
 */
public class WordCount extends Configured implements Tool {

    public static void main(String[] args) throws Exception {
        int res = ToolRunner.run(new Configuration(), new WordCount(), args);
        System.exit(res);
    }

    @Override
    public int run(String[] args) throws Exception {
        if (args.length < 3) {
            System.err.println("Usage: WordCount <task> <input> <output>");
            return -1;
        }

        Task task = Task.fromArgument(args[0]);
        if (task == null) {
            System.err.println("Unknown task '" + args[0] + "'. Expected one of: " + Task.listOptions());
            return -1;
        }

        Path input = new Path(args[1]);
        Path output = new Path(args[2]);

        switch (task) {
        case ERRORS_BY_HOUR:
            deleteIfExists(output);
            return runErrorsByHourJob(input, output);
        case TOP_ERROR_TAXIS:
            return runTopErrorTaxisPipeline(input, output);
        case FASTEST_TAXIS:
            deleteIfExists(output);
            return runFastestTaxisJob(input, output);
        default:
            return -1;
        }
    }

    private int runErrorsByHourJob(Path input, Path output) throws Exception {
        Job job = Job.getInstance(getConf(), "gps-errors-by-hour");
        job.setJarByClass(WordCount.class);

        job.setMapperClass(WordCountMapper.ErrorsByHourMapper.class);
        job.setMapOutputKeyClass(IntWritable.class);
        job.setMapOutputValueClass(IntWritable.class);

        job.setReducerClass(WordCountReducer.SumReducer.class);
        job.setOutputKeyClass(IntWritable.class);
        job.setOutputValueClass(IntWritable.class);

        FileInputFormat.addInputPath(job, input);
        FileOutputFormat.setOutputPath(job, output);

        return job.waitForCompletion(true) ? 0 : 1;
    }

    private int runTopErrorTaxisPipeline(Path input, Path output) throws Exception {
        Path intermediate = new Path(output.toString() + "_temp");
        deleteIfExists(intermediate);

        try {
            int firstJob = runTaxiErrorFractionJob(input, intermediate);
            if (firstJob != 0) {
                return firstJob;
            }

            deleteIfExists(output);
            return runTopErrorTaxisJob(intermediate, output);
        } finally {
            deleteIfExists(intermediate);
        }
    }

    private int runTaxiErrorFractionJob(Path input, Path output) throws Exception {
        Job job = Job.getInstance(getConf(), "taxi-gps-error-fractions");
        job.setJarByClass(WordCount.class);

        job.setMapperClass(WordCountMapper.TaxiErrorStatsMapper.class);
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(IntWritable.class);

        job.setReducerClass(WordCountReducer.TaxiErrorStatsReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(DoubleWritable.class);

        FileInputFormat.addInputPath(job, input);
        FileOutputFormat.setOutputPath(job, output);

        return job.waitForCompletion(true) ? 0 : 1;
    }

    private int runTopErrorTaxisJob(Path input, Path output) throws Exception {
        Job job = Job.getInstance(getConf(), "top-5-gps-error-taxis");
        job.setJarByClass(WordCount.class);

        job.setMapperClass(WordCountMapper.TopFractionMapper.class);
        job.setMapOutputKeyClass(NullWritable.class);
        job.setMapOutputValueClass(Text.class);

        job.setReducerClass(WordCountReducer.TopFractionReducer.class);
        job.setNumReduceTasks(1);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(DoubleWritable.class);

        FileInputFormat.addInputPath(job, input);
        FileOutputFormat.setOutputPath(job, output);

        return job.waitForCompletion(true) ? 0 : 1;
    }

    private int runFastestTaxisJob(Path input, Path output) throws Exception {
        Job job = Job.getInstance(getConf(), "top-10-fastest-taxis");
        job.setJarByClass(WordCount.class);

        job.setMapperClass(WordCountMapper.TaxiSpeedMapper.class);
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(WordCountMapper.DoublePairWritable.class);

        job.setReducerClass(WordCountReducer.TaxiSpeedReducer.class);
        job.setNumReduceTasks(1);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(DoubleWritable.class);

        FileInputFormat.addInputPath(job, input);
        FileOutputFormat.setOutputPath(job, output);

        return job.waitForCompletion(true) ? 0 : 1;
    }

    private void deleteIfExists(Path path) throws IOException {
        FileSystem fs = path.getFileSystem(getConf());
        if (fs.exists(path)) {
            fs.delete(path, true);
        }
    }

    private enum Task {
        ERRORS_BY_HOUR("errors-by-hour", "task1"),
        TOP_ERROR_TAXIS("top-error-taxis", "task2"),
        FASTEST_TAXIS("fastest-taxis", "task3");

        private final String[] aliases;

        Task(String... aliases) {
            this.aliases = aliases;
        }

        static Task fromArgument(String value) {
            if (value == null) {
                return null;
            }
            String normalized = value.trim().toLowerCase(Locale.US);
            if (normalized.isEmpty()) {
                return null;
            }
            for (Task task : values()) {
                if (task.name().toLowerCase(Locale.US).equals(normalized)) {
                    return task;
                }
                for (String alias : task.aliases) {
                    if (alias.equals(normalized)) {
                        return task;
                    }
                }
            }
            return null;
        }

        static String listOptions() {
            StringBuilder builder = new StringBuilder();
            for (Task task : values()) {
                if (builder.length() > 0) {
                    builder.append(", ");
                }
                builder.append(task.aliases[0]);
            }
            return builder.toString();
        }
    }
}

