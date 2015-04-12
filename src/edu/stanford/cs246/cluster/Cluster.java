package edu.stanford.cs246.cluster;

import java.io.*;
import java.util.*;
import java.lang.Math;
import java.lang.System;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

public class Cluster extends Configured implements Tool {

   private static double[][] dataPoints;
   private static double[][] clusterPoints;
   private static HashSet<Integer> usedClusters;
   private static String[] tagNames;
   private static double phi = 0;
   private static int numDataPoints = 4601;
   private static int numClusterPoints = 10;
   private static int numDimensions = 58;
   private static int numIterations = 20;
   private static int doc2Cluster = 0;

   public class FilterScore implements Comparator<FilterScore>, Comparable<FilterScore>{
		public int id;
		public double score;

		public FilterScore(int id_num, double filter_score){
			id = id_num;
			score = filter_score;
		}

		public String toString(){
			String self_name = tagNames[id];
			return self_name + ":" + score;
		}

		//Overriding the compareTo method
		public int compareTo(FilterScore fs){
			Double thisScore = new Double(this.score);
			Double thatScore = new Double(fs.score);
			int comp = thisScore.compareTo(thatScore);
			if(comp == 0.0){
				if(this.id < fs.id){
					return -1;
				}else{
					return 1;
				}
			}else{
				return comp;
			}
		}

		//Overriding the compare method
		public int compare(FilterScore fs1, FilterScore fs2){
			double diff = fs1.score - fs2.score;
			if(diff == 0){
				if(fs1.id < fs2.id){
					return -1;
				}else{
					return 1;
				}
			}else{
				if(diff < 0){
					return -1;
				}else{
					return 1;
				}
			}
		}

	}
   
   public static void main(String[] args) throws Exception {
      System.out.println(Arrays.toString(args));
      int res = ToolRunner.run(new Configuration(), new Cluster(), args);
      System.exit(res);
   }
   
   @Override
   public int run(String[] args) throws Exception {
	  String randomClusterFile = args[0];
	  String strategicClusterFile = args[1];
	  String dataFile = args[2];
	  String vocabFile = args[3];
	  String outputFile = args[4];
	  String indexFile = "dataIndex.txt";
	  String clusterInput = randomClusterFile;
	  dataPoints = new double[numDataPoints][numDimensions];
	  clusterPoints = new double[numClusterPoints][numDimensions];
	  tagNames = new String[numDimensions];
	  populateTags(vocabFile,tagNames);
	  readData(dataFile,dataPoints);
	  indexData(dataPoints,indexFile);
	  System.out.println(Arrays.toString(args));
	  //We run for the max iterations and do not stop if it converges
	  ArrayList<Double> errorByIteration = new ArrayList<Double>(numIterations);
	  for(int i = 0; i < numIterations; i++){
		  phi = 0;
		  usedClusters = new HashSet<Integer>(numClusterPoints);
		  String file = clusterInput;
		  if(i != 0) file = file + "/part-r-00000";
		  readData(file,clusterPoints);
		  clusterInput = outputFile + i;
	      Job job = new Job(getConf(), "Cluster");
	      job.setJarByClass(Cluster.class);
	      job.setOutputKeyClass(Text.class);
	      job.setOutputValueClass(IntWritable.class);
	      job.setMapperClass(Map.class);
	      job.setReducerClass(Reduce.class);
	      job.setInputFormatClass(TextInputFormat.class);
	      job.setOutputFormatClass(TextOutputFormat.class);
	      FileInputFormat.addInputPath(job, new Path(indexFile));
	      FileOutputFormat.setOutputPath(job, new Path(clusterInput));
	      job.waitForCompletion(true);
	      addUnusedClusters(usedClusters,clusterInput);
	      errorByIteration.add(new Double(phi));
	      //System.out.println("iteration: " + i + " phi: " + phi);
	  }
	  System.out.println("Document 2 is in cluster: " + doc2Cluster);
	  printError(errorByIteration);
	  ArrayList<FilterScore> fs = convertArray(clusterPoints[doc2Cluster]);
	  Collections.sort(fs);
	  Collections.reverse(fs);
	  System.out.println("Printing out the top 10 tags for document 2");
	  for(int i = 0; i < 10; i++){
		  System.out.println(fs.get(i));
	  }
      return 0;
   }
   
   private ArrayList<FilterScore> convertArray(double[] ar){
	   int size = ar.length;
	   ArrayList<FilterScore> fsAr = new ArrayList<FilterScore>(size);
	   for(int i = 0; i < size; i++){
		   FilterScore fs = new FilterScore(i,ar[i]);
		   fsAr.add(fs);
	   }
	   return fsAr;
   }
   
   private static void populateTags(String file, String[] tags){
		try{
			BufferedReader br = new BufferedReader(new FileReader(file));
			String tagName = br.readLine();
			int i = 0;
			while(tagName != null){
				tags[i] = tagName;
				i = i + 1;
				tagName = br.readLine();
			}
			br.close();
		}catch(IOException e){
			System.out.println("Error reading in " + file);
			e.printStackTrace();
		}
   }
   
   private void printError(ArrayList<Double> err){
	   String file = "phiError.txt";
		try{
			Writer writer = new BufferedWriter(new OutputStreamWriter(
				new FileOutputStream(file), "utf-8"));
			int size = err.size();
			for(int i = 0; i < size; i++){
				writer.write(err.get(i) + "\n");
			}
			writer.close();
		}catch(IOException e){
			System.out.println("Error printing out phi " + file);
			e.printStackTrace();
		}
   }
   
   private static void addUnusedClusters(HashSet<Integer> usedClusters, String outputFile){
	   String file = outputFile + "/part-r-00000";
		try{
			Writer writer = new BufferedWriter(new OutputStreamWriter(
				new FileOutputStream(file,true), "utf-8"));
			int size = clusterPoints.length;
			for(int i = 0; i < size; i++){
				if(!usedClusters.contains(new Integer(i))){
					String vector = getVectorString(clusterPoints[i]);
					writer.write(vector + "\n");
				}
			}
			writer.close();
		}catch(IOException e){
			System.out.println("Error adding unused vectors " + file);
			e.printStackTrace();
		}
   }
   
   private static double[] centroidVector(double[][] vectors, ArrayList<Integer> indices){
	   double[] centroid = new double[numDimensions];
	   int numVectors = indices.size();
	   if(numVectors == 0) return centroid;
	   for(int i = 0; i < numDimensions; i++){
		   double value = 0.0;
		   for(int j = 0; j < numVectors; j++){
			   int index = indices.get(j);
			   value = value + vectors[index][i];
		   }
		   value = value/(double)numVectors;
		   centroid[i] = value;
	   }
	   return centroid;
   }
   
   private static double distanceBetween(double[] vector1, double[] vector2){
	   double distance = 0.0;
	   for(int i = 0; i < numDimensions; i++){
		   double sum = vector1[i] - vector2[i];
		   sum = sum*sum;
		   distance = distance + sum;
	   }
	   return Math.sqrt(distance);
   }

   private static void indexData(double[][] dataPoints, String file){
		try{
			Writer writer = new BufferedWriter(new OutputStreamWriter(
				new FileOutputStream(file), "utf-8"));
			int size = dataPoints.length;
			for(int i = 0; i < size; i++){
				writer.write(i + "\n");
			}
			writer.close();
		}catch(IOException e){
			System.out.println("Error writing to " + file);
			e.printStackTrace();
		}
   }
   
   private static String getVectorString(double[] ar){
	   String vector = "";
       for(int i = 0; i < ar.length; i++){
      	 vector = vector + ar[i] + " ";
       }
       return vector;
   }
   
   private static void readData(String file, double[][] ar){
		try{
			BufferedReader br = new BufferedReader(new FileReader(file));
			String dataPoint = br.readLine();
			int i = 0;
			while(dataPoint != null){
				StringTokenizer cols = new StringTokenizer(dataPoint);
				int j = 0;
				while(cols.hasMoreTokens()){
					ar[i][j] = Double.parseDouble(cols.nextToken());
					j++;
				}
				dataPoint = br.readLine();
				i++;
			}
			br.close();
		}catch(IOException e){
			System.out.println("Error reading in " + file);
			e.printStackTrace();
		}
   }
   
   public static class Map extends Mapper<LongWritable, Text, Text, IntWritable> {
      private Text outputKey = new Text("");
      private final static IntWritable outputValue = new IntWritable(1);

      @Override
      public void map(LongWritable key, Text value, Context context)
              throws IOException, InterruptedException {
    	 for (String token: value.toString().split("\\s+")) {
	    	 int dataPoint = Integer.parseInt(token);
	    	 int cluster = closestCluster(dataPoint);
	    	 outputValue.set(dataPoint);
	    	 outputKey.set(cluster + "");
	    	 context.write(outputKey, outputValue);
	    	 //System.out.println(outputKey.toString());
    	 }
      }
      
      private int closestCluster(int dataID){
    	  double lowestDistance = Double.MAX_VALUE;
    	  int clusterPoint = 0;
    	  for(int i = 0; i < numClusterPoints; i++){
    		  double score = distanceBetween(dataPoints[dataID],clusterPoints[i]);
    		  if(score < lowestDistance){
    			  lowestDistance = score;
    			  clusterPoint = i;
    		  }
    	  }
    	  return clusterPoint;
      }
   }

   public static class Reduce extends Reducer<Text, IntWritable, Text, Text> {
	  private Text outputKey = new Text("");
	  private Text outputValue = new Text("");
	  
      @Override
      public void reduce(Text key, Iterable<IntWritable> values, Context context)
              throws IOException, InterruptedException {
    	 int clusterID = Integer.parseInt(key.toString());
         ArrayList<Integer> indices = new ArrayList<Integer>();
         for (IntWritable val : values) {
        	 int value = val.get();
        	 indices.add(new Integer(value));
        	 //We need to know what cluster document 2 is in to answer the question
        	 if(value == 1) doc2Cluster = clusterID;
         }
         double error = getError(clusterID,indices,clusterPoints,dataPoints);
         phi = phi + error;
         //Track the clusters that we have used so far in case some clusters die out
         usedClusters.add(new Integer(clusterID));
         double[] centroid = centroidVector(dataPoints,indices);
         outputKey.set("");
         outputValue.set(getVectorString(centroid));
         //System.out.println(outputValue.toString());
         context.write(outputKey, outputValue);
      }
      
      private static double getError(int clusterIndex, ArrayList<Integer> points, double[][] clusters, double[][] dataPoints){
    	 double error = 0;
    	 int numVectors = points.size();
    	 for(int i = 0; i < numVectors; i++){
    		 double diff = 0;
    		 diff = distanceBetween(clusters[clusterIndex],dataPoints[points.get(i)]);
    		 error = error + diff*diff;
    	 }
    	 return error;
      }
   }
}
