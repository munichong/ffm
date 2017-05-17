package ffm;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.LinkedList;

/**
 * @author chenhuang
 *
 */



public class FFMProblem {
	//data : field_num:feature_num:value
	// max(feature_num) + 1
	public int n; // the total number of features in the dataset
	// max(field_num) + 1
	public int m; // the total number of fields in the dataset
	public int l;
	// X[ [P[0], P[1]) ], length=nnz
	public FFMNode[] X;
	// length=l+1
	public int[] P;
	// Y[0], length=l
	public float[] Y;
	
	public static int MIN_DWELL_TIME = 1;
	public static boolean SHUFFLE_TRAINING = true;
	
	
	public static FFMProblem readFFMProblem(String path) throws IOException {
		FFMProblem problem = new FFMProblem();
		
		/*
		 * Iterate through the data
		 * Determine l and nnz
		 */
		int l = 0; // The number of training instances
		int nnz = 0; // Number of Non- Zero entries?  the number of non-zero elements of each impression
//		System.out.println(path);
		BufferedReader br = new BufferedReader(new InputStreamReader(
				new FileInputStream(new File(path)), "UTF-8"));
		
		LinkedList<String> input_data = new LinkedList<String>();
		String line = null;
//		br.readLine();
		while((line = br.readLine()) != null) { // in order to get l and nnz
			l += 1;
			input_data.add(line);
			String[] fields = line.split(" |\t"); // !!! Regex: split by " " or "\t"
			for(int i=1; i<fields.length; i++) {
				nnz += 1;
			}
		}
		br.close();
		
		System.out.printf("reading %s, instance_num: %d, nnz: %d\n", path, l, nnz);
		
//		/*
//		 * Shuffle the data
//		 */
//		if(SHUFFLE_TRAINING) {
//			System.out.println("Shuffling training data");
//			BufferedWriter bw = new BufferedWriter(new FileWriter(new File(path + ".tmp")));
//			for(int i=0; i<l; i++) {
//				System.out.println(i);
//		        int card = (int) (Math.random() * (l-i));
//		        bw.write(input_data.remove(card) + "\n");
//		    }
////			for(String instance : input_data) {
////				bw.write(instance + "\n");
////			}
//		}
		
		
		/*
		 * Officially Build the data in the memory
		 */
		problem.l = l;
		problem.X = new FFMNode[nnz];
		problem.Y = new float[l];
		problem.P = new int[l+1];
		problem.P[0] = 0;
		
		br = new BufferedReader(new InputStreamReader(
				new FileInputStream(new File(path)), "UTF-8"));
		int p = 0;
//		br.readLine();
		for(int i=0; (line=br.readLine())!=null; i++) {
			String[] fields = line.split(" |\t");
			
			/**
			 * CLASSIFICATION
			 */
			problem.Y[i] = (Integer.parseInt(fields[0]) > MIN_DWELL_TIME)?1.f:-1.f; // The target is 1 or -1
			
			for(int j=1; j<fields.length; j++) {
				String[] subFields = fields[j].split(":");
				FFMNode node = new FFMNode(); // each non-zero cell is a node
				node.f = Integer.parseInt(subFields[0]); // field index
				node.j = Integer.parseInt(subFields[1]); // feature index
				node.v = Float.parseFloat(subFields[2]); // value
				problem.X[p] = node;
				problem.m = Math.max(problem.m, node.f + 1); // the total number of fields in the dataset
				problem.n = Math.max(problem.n, node.j + 1); // the total number of features in the dataset
				p++;
			}
			problem.P[i+1] = p; // The cumulative number of fields so far (i.e. the number of cells (zero or non-zero) in this training instance.)
		}
		br.close();
		
		return problem;
	}

	@Override
	public String toString() {
		return "FFMProblem [n=" + n + ", m=" + m + ", l=" + l + ", X="
				+ Arrays.toString(X) + ", P=" + Arrays.toString(P) + ", Y="
				+ Arrays.toString(Y) + "]";
	}
	
//	public static void main(String[] args) throws IOException {
//		FFMProblem problem = FFMProblem.readFFMProblem("aaa");
//		System.out.println(problem);
//	}
	
}
