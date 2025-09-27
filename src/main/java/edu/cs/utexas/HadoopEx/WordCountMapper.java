package edu.cs.utexas.HadoopEx;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Comparator;
import java.util.PriorityQueue;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Mapper;

/**
 * Collection of mapper implementations used by the assignment MapReduce jobs.
 */
public final class WordCountMapper {

    private WordCountMapper() {
    }

    /**
     * Lightweight representation of a taxi trip extracted from a CSV row.
     */
    static final class TaxiTripRecord {
        final String taxiId;
        final int pickupHour; // 1-24, -1 if unavailable
        final boolean hasGpsError;
        final double distanceMiles;
        final double durationMinutes;

        private TaxiTripRecord(String taxiId, int pickupHour, boolean hasGpsError, double distanceMiles,
                double durationMinutes) {
            this.taxiId = taxiId;
            this.pickupHour = pickupHour;
            this.hasGpsError = hasGpsError;
            this.distanceMiles = distanceMiles;
            this.durationMinutes = durationMinutes;
        }

        static TaxiTripRecord parse(String line) {
            if (line == null) {
                return null;
            }

            String[] fields = line.split(",");
            if (fields.length < 17) {
                return null;
            }

            for (int i = 0; i < fields.length; i++) {
                fields[i] = fields[i].trim();
            }

            String taxiId = fields[0];
            if (taxiId.isEmpty()) {
                return null;
            }

            int pickupHour = parseHour(fields[2]);
            boolean gpsError = isGpsError(fields[6]) || isGpsError(fields[7]) || isGpsError(fields[8])
                    || isGpsError(fields[9]);

            double tripSeconds = parseDouble(fields[4]);
            double distance = parseDouble(fields[5]);
            double durationMinutes = Double.isNaN(tripSeconds) ? Double.NaN : tripSeconds / 60.0d;

            return new TaxiTripRecord(taxiId, pickupHour, gpsError, distance, durationMinutes);
        }

        boolean hasPickupHour() {
            return pickupHour > 0;
        }

        boolean hasValidSpeedData() {
            return !Double.isNaN(distanceMiles) && !Double.isNaN(durationMinutes) && durationMinutes > 0.0d;
        }
    }

    /**
     * Writable pair used to pass distance and duration information.
     */
    public static final class DoublePairWritable implements Writable {
        private double first;
        private double second;

        public DoublePairWritable() {
        }

        public DoublePairWritable(double first, double second) {
            this.first = first;
            this.second = second;
        }

        double getFirst() {
            return first;
        }

        double getSecond() {
            return second;
        }

        @Override
        public void write(DataOutput out) throws IOException {
            out.writeDouble(first);
            out.writeDouble(second);
        }

        @Override
        public void readFields(DataInput in) throws IOException {
            first = in.readDouble();
            second = in.readDouble();
        }
    }

    private static double parseDouble(String value) {
        if (value == null) {
            return Double.NaN;
        }
        String trimmed = value.trim();
        if (trimmed.isEmpty()) {
            return Double.NaN;
        }
        try {
            double parsed = Double.parseDouble(trimmed);
            if (Double.isNaN(parsed) || Double.isInfinite(parsed) || parsed < 0.0d) {
                return Double.NaN;
            }
            return parsed;
        } catch (NumberFormatException ex) {
            return Double.NaN;
        }
    }

    private static boolean isGpsError(String value) {
        if (value == null) {
            return true;
        }
        String trimmed = value.trim();
        if (trimmed.isEmpty()) {
            return true;
        }
        try {
            return Double.parseDouble(trimmed) == 0.0d;
        } catch (NumberFormatException ex) {
            return true;
        }
    }

    private static int parseHour(String pickupDateTime) {
        if (pickupDateTime == null) {
            return -1;
        }
        String trimmed = pickupDateTime.trim();
        if (trimmed.isEmpty()) {
            return -1;
        }

        String timeComponent = trimmed;
        int spaceIdx = timeComponent.indexOf(' ');
        int tIdx = timeComponent.indexOf('T');
        if (spaceIdx >= 0) {
            timeComponent = timeComponent.substring(spaceIdx + 1);
        } else if (tIdx >= 0) {
            timeComponent = timeComponent.substring(tIdx + 1);
        }

        if (!timeComponent.contains(":")) {
            return -1;
        }
        String[] timePieces = timeComponent.split(":");
        if (timePieces.length == 0 || timePieces[0].isEmpty()) {
            return -1;
        }
        try {
            int hour = Integer.parseInt(timePieces[0]);
            if (hour < 0 || hour > 23) {
                return -1;
            }
            return (hour % 24) + 1;
        } catch (NumberFormatException ex) {
            return -1;
        }
    }

    /** Mapper for Task 1: emit the pickup hour for rows with GPS errors. */
    public static final class ErrorsByHourMapper extends Mapper<LongWritable, Text, IntWritable, IntWritable> {
        private static final IntWritable ONE = new IntWritable(1);
        private final IntWritable hourWritable = new IntWritable();

        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            TaxiTripRecord record = TaxiTripRecord.parse(value.toString());
            if (record == null) {
                return;
            }

            if (record.hasGpsError && record.hasPickupHour()) {
                hourWritable.set(record.pickupHour);
                context.write(hourWritable, ONE);
            }
        }
    }

    /** Mapper for Task 2: track GPS errors per taxi. */
    public static final class TaxiErrorStatsMapper extends Mapper<LongWritable, Text, Text, IntWritable> {
        private static final IntWritable ERROR = new IntWritable(1);
        private static final IntWritable NO_ERROR = new IntWritable(0);
        private final Text taxiKey = new Text();

        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            TaxiTripRecord record = TaxiTripRecord.parse(value.toString());
            if (record == null) {
                return;
            }

            taxiKey.set(record.taxiId);
            context.write(taxiKey, record.hasGpsError ? ERROR : NO_ERROR);
        }
    }

    /** Mapper for Task 2 (job 2): keep a local top-5 queue of error fractions. */
    public static final class TopFractionMapper extends Mapper<LongWritable, Text, NullWritable, Text> {
        private static final int LIMIT = 5;
        private final PriorityQueue<WordCountReducer.MetricEntry> queue =
                new PriorityQueue<>(Comparator.comparingDouble(entry -> entry.value));

        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString().trim();
            if (line.isEmpty()) {
                return;
            }

            String[] parts = line.split("\\t");
            if (parts.length != 2) {
                return;
            }

            try {
                double fraction = Double.parseDouble(parts[1]);
                WordCountReducer.MetricEntry entry = new WordCountReducer.MetricEntry(parts[0], fraction);
                queue.add(entry);
                if (queue.size() > LIMIT) {
                    queue.poll();
                }
            } catch (NumberFormatException ex) {
                // ignore malformed lines
            }
        }

        @Override
        protected void cleanup(Context context) throws IOException, InterruptedException {
            for (WordCountReducer.MetricEntry entry : queue) {
                context.write(NullWritable.get(), new Text(entry.id + "\t" + entry.value));
            }
        }
    }

    /** Mapper for Task 3: emit distance/time tuples per taxi. */
    public static final class TaxiSpeedMapper extends Mapper<LongWritable, Text, Text, DoublePairWritable> {
        private final Text taxiKey = new Text();

        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            TaxiTripRecord record = TaxiTripRecord.parse(value.toString());
            if (record == null || record.taxiId.isEmpty()) {
                return;
            }

            if (!record.hasValidSpeedData()) {
                return;
            }

            taxiKey.set(record.taxiId);
            context.write(taxiKey, new DoublePairWritable(record.distanceMiles, record.durationMinutes));
        }
    }
}

