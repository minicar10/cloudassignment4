package edu.cs.utexas.HadoopEx;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

/**
 * Collection of reducer implementations used by the assignment MapReduce jobs.
 */
public final class WordCountReducer {

    private WordCountReducer() {
    }

    /** Simple value/count pair used for ranking outputs. */
    public static final class MetricEntry {
        public final String id;
        public final double value;

        public MetricEntry(String id, double value) {
            this.id = id;
            this.value = value;
        }
    }

    private static PriorityQueue<MetricEntry> newQueue() {
        return new PriorityQueue<>(Comparator.comparingDouble(entry -> entry.value));
    }

    /** Reducer for Task 1: sum error counts per hour. */
    public static final class SumReducer extends Reducer<IntWritable, IntWritable, IntWritable, IntWritable> {
        private final IntWritable result = new IntWritable();

        @Override
        protected void reduce(IntWritable key, Iterable<IntWritable> values, Context context)
                throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable value : values) {
                sum += value.get();
            }
            result.set(sum);
            context.write(key, result);
        }
    }

    /** Reducer for Task 2 job 1: compute GPS error fractions per taxi. */
    public static final class TaxiErrorStatsReducer extends Reducer<Text, IntWritable, Text, DoubleWritable> {
        private final DoubleWritable fractionWritable = new DoubleWritable();

        @Override
        protected void reduce(Text key, Iterable<IntWritable> values, Context context)
                throws IOException, InterruptedException {
            int total = 0;
            int errors = 0;
            for (IntWritable value : values) {
                total++;
                if (value.get() > 0) {
                    errors++;
                }
            }

            if (total == 0) {
                return;
            }

            double fraction = (double) errors / (double) total;
            fractionWritable.set(fraction);
            context.write(key, fractionWritable);
        }
    }

    /** Reducer for Task 2 job 2: retain global top-5 error fractions. */
    public static final class TopFractionReducer extends Reducer<NullWritable, Text, Text, DoubleWritable> {
        private static final int LIMIT = 5;
        private final PriorityQueue<MetricEntry> queue = newQueue();
        private final DoubleWritable result = new DoubleWritable();

        @Override
        protected void reduce(NullWritable key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {
            for (Text value : values) {
                String[] parts = value.toString().split("\\t");
                if (parts.length != 2) {
                    continue;
                }

                try {
                    double fraction = Double.parseDouble(parts[1]);
                    queue.add(new MetricEntry(parts[0], fraction));
                    if (queue.size() > LIMIT) {
                        queue.poll();
                    }
                } catch (NumberFormatException ex) {
                    // Ignore malformed lines
                }
            }
        }

        @Override
        protected void cleanup(Context context) throws IOException, InterruptedException {
            List<MetricEntry> entries = new ArrayList<>(queue);
            entries.sort((a, b) -> Double.compare(b.value, a.value));
            for (MetricEntry entry : entries) {
                result.set(entry.value);
                context.write(new Text(entry.id), result);
            }
        }
    }

    /** Reducer for Task 3: compute the global top-10 fastest taxis. */
    public static final class TaxiSpeedReducer
            extends Reducer<Text, WordCountMapper.DoublePairWritable, Text, DoubleWritable> {
        private static final int LIMIT = 10;
        private final PriorityQueue<MetricEntry> queue = newQueue();
        private final DoubleWritable result = new DoubleWritable();

        @Override
        protected void reduce(Text key, Iterable<WordCountMapper.DoublePairWritable> values, Context context)
                throws IOException, InterruptedException {
            double distance = 0.0d;
            double minutes = 0.0d;

            for (WordCountMapper.DoublePairWritable value : values) {
                distance += value.getFirst();
                minutes += value.getSecond();
            }

            if (minutes <= 0.0d) {
                return;
            }

            double milesPerMinute = distance / minutes;
            queue.add(new MetricEntry(key.toString(), milesPerMinute));
            if (queue.size() > LIMIT) {
                queue.poll();
            }
        }

        @Override
        protected void cleanup(Context context) throws IOException, InterruptedException {
            List<MetricEntry> entries = new ArrayList<>(queue);
            entries.sort((a, b) -> Double.compare(b.value, a.value));
            for (MetricEntry entry : entries) {
                result.set(entry.value);
                context.write(new Text(entry.id), result);
            }
        }
    }
}

