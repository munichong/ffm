package ffm;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.Reader;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.Hashtable;
import java.util.LinkedList;
import java.util.Random;

public class FFMModel {
	// max(feature_num) + 1
	public int n;
	// max(field_num) + 1
	public int m;
	// latent factor dim
	public int k;
	// length = n * m * k * 2
	public float[] W;
	public boolean normalization;
	
	
	public static boolean hrchyReg;
	public static float C;
	public static HierarchicalRegularization hr;
	
	
	public FFMModel initModel(int n_, int m_, FFMParameter param) throws IOException {
		n = n_; // the total number of features 
		m = m_; // the total number of fields
		k = param.k; // the dimensionality
		normalization = param.normalization;
		hrchyReg = param.hrchyReg;
		C = param.C;
		W = new float[n * m * k * 2];
		
		float coef = (float) (0.5 / Math.sqrt(k));
		Random random = new Random();
		
		int position = 0;
		for (int j = 0; j < n; j++) {
			for(int f = 0; f < m; f++) {
				for(int d = 0; d < k; d++) {
					// store w_{j1,f2}
					W[position] = coef * random.nextFloat();
					position += 1;
				}
				for(int d = this.k; d < 2*this.k; d++) {
					// store (G_{j1,f2}), i.e., the sum of squared gradient
					W[position] = 1.f;
					position += 1;
				}
			}
		}
		
		if(hrchyReg) {
			hr = new HierarchicalRegularization(m, k);
			// increment the section and channel sum vectors with the above random W vectors.
//			initSumVectors(hr.sect_sum_vec);
//			initSumVectors(hr.chan_sum_vec);
		}
		
		return this;
	}
	
	private void initSumVectors(Hashtable<Integer, float[][]> sum_vec) {
		for(Enumeration<Integer> en = sum_vec.keys(); en.hasMoreElements(); ) {
			int feature_index = en.nextElement();
			float[][] vector_arr = sum_vec.get(feature_index);
			for(int f = 1; f < m; f++) {
				for(int d = 0; d < k; d++) {
					vector_arr[f][d] = W[feature_index * m * k * 2 + f * k * 2 + d];
				}
			}
			sum_vec.put(feature_index, vector_arr);
		}
	}

	public void saveModel(String path) throws IOException {
		BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(
				new FileOutputStream(new File(path)), "UTF-8"));
		bw.write("n " + n + "\n");
		bw.write("m " + m + "\n");
		bw.write("k " + k + "\n");
		bw.write("normalization " + normalization + "\n");
		int align0 = k * 2;
		int align1 = m * k * 2;
		for(int j=0; j<n; j++) {
			for(int f=0; f<m; f++) {
				bw.write("w" + j + "," + f + " ");
				for(int d = 0; d<k; d++) {
					bw.write(W[j*align1 + f*align0 + d] + " ");
				}
				bw.write("\n");
			}
		}
		bw.close();
	}
	
	public static FFMModel loadModel(String path) throws IOException {
		FFMModel model = new FFMModel();
		BufferedReader br = new BufferedReader(new InputStreamReader(
				new FileInputStream(new File(path)), "UTF-8"));
		model.n = Integer.parseInt(br.readLine().split(" ")[1]);
		model.m = Integer.parseInt(br.readLine().split(" ")[1]);
		model.k = Integer.parseInt(br.readLine().split(" ")[1]);
		model.normalization = Boolean.parseBoolean(br.readLine().split(" ")[1]);
		model.W = new float[model.n * model.m * model.k * 2];
		int align0 = model.k * 2;
		int align1 = model.m * model.k * 2;
		for(int j = 0; j < model.n; j++) {
			for(int f = 0; f < model.m; f++) {
				String line = br.readLine().trim();
				String[] fields = line.split(" ");
				for(int d = 0; d < model.k; d++) {
					model.W[j*align1 + f*align0 + d] = Float.parseFloat(fields[1+d]);
				}
			}
		}
		br.close();
		return model;
	}
	
	public static float[] normalize(FFMProblem problem, boolean normal) {
		float[] R = new float[problem.l];
		if(normal) {
			for(int i=0; i<problem.l; i++) {
				double norm = 0;
				for(int p=problem.P[i]; p<problem.P[i+1]; p++) {
					norm += problem.X[p].v * problem.X[p].v;
				}
				R[i] = (float) (1.f / norm);
			}
		} else {
			for(int i=0; i<problem.l; i++) {
				R[i] = 1.f;
			}
		}		
		return R;
	}
	
	public static int[] randomization(int l, boolean rand) {
		int[] order = new int[l];
		for (int i = 0; i < order.length; i++) {
			order[i] = i;
		}
		if(rand) {
			Random random = new Random();
			for(int i = order.length; i > 1; i--) {
				int tmp = order[i-1];
				int index = random.nextInt(i);
				order[i-1] = order[index];
				order[index] = tmp;
			}
		}
		return order;
	}
	
	public static float wTx(FFMProblem prob, int i, float r, FFMModel model,
			float kappa, float eta, float lambda, boolean do_update) {
		// kappa = -y * exp(-y*t) / (1+exp(-y*t))
		int start = prob.P[i]; // the start index of this instance in Problem.X; i represents the random training instance
		int end = prob.P[i+1]; // the end index of this instance in Problem.X
		float t = 0.f;
		int align0 = model.k * 2; // dimensionality * 2
		int align1 = model.m * model.k * 2; // #fields * dimensionality * 2
		
		for(int N1 = start; N1 < end; N1++) { // iterate through all nodes of this instance i 
			// N1 is the index of a node of the instance i
			int j1 = prob.X[N1].j; // j is the feature index
			int f1 = prob.X[N1].f; // f is the field index
			float v1 = prob.X[N1].v; // v is the value
			if(j1 >= model.n || f1 >= model.m) 
				continue;
			
			for(int N2 = N1+1; N2 < end; N2++) {
				// N1 is the index of a node of the instance i other than N1
				int j2 = prob.X[N2].j; // j is the feature index
				int f2 = prob.X[N2].f; // f is the field index
				
				float v2 = prob.X[N2].v;
				if(j2 >= model.n || f2 >= model.m) 
					continue;
				
				int w1_index = j1 * align1 + f2 * align0; // wi_index is used to locate the latent vector in W
				int w2_index = j2 * align1 + f1 * align0;
				float v = 2.f * v1 * v2 * r;
				
				if(do_update) {
					int wg1_index = w1_index + model.k;
					int wg2_index = w2_index + model.k;
					float kappav = kappa * v;
					for(int d = 0; d < model.k; d++) {
						// ============== update the gradient sum ==============
						float g1 = 0;
						if(hrchyReg && f1 == 2) { // if f1 is page
//							g1 = lambda * model.W[w1_index+d] + kappav * model.W[w2_index+d] + C * model.W[w1_index+d];
							g1 = lambda * model.W[w1_index+d] + kappav * model.W[w2_index+d];
							
							float meanParSectsW = 0;
							if(!hr.page2sect.containsKey(j1) && j1 != 169)
								System.out.println(j1 + " not in page2sect");
							if(hr.page2sect.containsKey(j1)) {
								for(int sect_index : hr.page2sect.get(j1)) {
									meanParSectsW += model.W[sect_index * align1 + f2 * align0 + d];
								}
								g1 -= C * meanParSectsW / hr.page2sect.get(j1).size();
							}
							
							
//							float meanParChansW = 0;
//							for(int chan_index : hr.page2chan.get(j1)) {
//								meanParChansW += model.W[chan_index * align1 + f2 * align0];
//							}
//							g1 -= C * meanParChansW / hr.page2chan.get(j1).size();
						}
//						else if (hrchyReg && f1 == 4) { // if f1 is channel
//							g1 = lambda * model.W[w1_index + d] + kappav * model.W[w2_index + d] + C * model.W[w1_index + d];
//						}
						else if(hrchyReg && f1 == 5) { // if f1 is section
//							g1 = lambda * model.W[w1_index + d] + kappav * model.W[w2_index + d] + C * model.W[w1_index + d];
							g1 = lambda * model.W[w1_index + d] + kappav * model.W[w2_index + d];
							
							float meanParChansW = 0;
							for(int chan_index : hr.sect2chan.get(j1)) {
								meanParChansW += model.W[chan_index * align1 + f2 * align0 + d];
							}
							g1 -= C * meanParChansW / hr.sect2chan.get(j1).size();
						}
						else {
							g1 = lambda * model.W[w1_index+d] + kappav * model.W[w2_index+d];
						}
						
						
						
						float g2 = 0;
//						if(hrchyReg && f2 == 2 && hr.page2sect.containsKey(j2)) { // if f2 is page
						if(hrchyReg && f2 == 2) { // if f2 is page
//							g2 = lambda * model.W[w2_index + d] + kappav * model.W[w1_index + d] + C * model.W[w2_index + d];
							g2 = lambda * model.W[w2_index + d] + kappav * model.W[w1_index + d];
							
							float meanParSectsW = 0;
							if(!hr.page2sect.containsKey(j2) && j2 != 169)
								System.out.println(j2 + " not in page2sect");
							if(hr.page2sect.containsKey(j2)) {
								for(int page_index : hr.page2sect.get(j2)) {
									meanParSectsW += model.W[page_index * align1 + f1 * align0 + d];
								}
								g2 -= C * meanParSectsW / hr.page2sect.get(j2).size();
							}
							
							
//							float meanParChansW = 0;
//							for(int chan_index : hr.page2chan.get(j2)) {
//								meanParChansW += model.W[chan_index * align1 + f1 * align0];
//							}
//							g2 -= C * meanParChansW / hr.page2chan.get(j2).size();
						}
//						else if (hrchyReg && f2 == 4) { // if f1 is channel
//							g2 = lambda * model.W[w2_index + d] + kappav * model.W[w1_index + d] + C * model.W[w2_index + d];
//						}
						else if(hrchyReg && f2 == 5) { // if f2 is section
//							g2 = lambda * model.W[w2_index + d] + kappav * model.W[w1_index + d] + C * model.W[w2_index + d];
							g2 = lambda * model.W[w2_index + d] + kappav * model.W[w1_index + d];
							
							float meanParChansW = 0;
							if(!hr.sect2chan.containsKey(j2) && j2 != 144)
								System.out.println(j2 + " is not in sect2chan.");
							if(hr.sect2chan.containsKey(j2)) {
								for(int chan_index : hr.sect2chan.get(j2)) {
									meanParChansW += model.W[chan_index * align1 + f1 * align0 + d];
								}
							    g2 -= C * meanParChansW / hr.sect2chan.get(j2).size();
							}
						}
						else {
							g2 = lambda * model.W[w2_index + d] + kappav * model.W[w1_index + d];
						}
						
						
						
							
						float wg1 = model.W[wg1_index+d] + g1 * g1;
						float wg2 = model.W[wg2_index+d] + g2 * g2;
							
						// ============== update model ==============
						model.W[w1_index+d] = model.W[w1_index+d] - eta / (float)(Math.sqrt(wg1)) * g1;
						model.W[w2_index+d] = model.W[w2_index+d] - eta / (float)(Math.sqrt(wg2)) * g2;
							
						model.W[wg1_index+d] = wg1;
						model.W[wg2_index+d] = wg2;
						// =====================================
						
					}
				} else {
					for(int d = 0; d < model.k; d++) {
						t += model.W[w1_index + d] * model.W[w2_index + d] * v;
					}
				}
			}
		}	
		
//		// update the sum vector
//		if(hrchyReg) {
//			
//		}
		
		return t;
	}
	
	public static FFMModel train(FFMProblem tr, FFMProblem va, FFMParameter param) throws IOException {
		FFMModel model = new FFMModel();
		model.initModel(tr.n, tr.m, param);
		
		float[] R_tr = normalize(tr, param.normalization);
		float[] R_va = null;	
		if(va != null) {
			R_va = normalize(va, param.normalization);
		}
			
		for(int iter = 0; iter < param.n_iters; iter++) { // epochs
			double tr_loss = 0.;
			// randomize the training input at the beginning of every iteration
			int[] order = randomization(tr.l, param.random);
			for(int ii=0; ii<tr.l; ii++) {
				int i = order[ii];
				float y = tr.Y[i];
				float r = R_tr[i];
				float t = wTx(tr, i, r, model, 0.f, 0.f, 0.f, false);
				float expnyt = (float) Math.exp(-y * t);
				tr_loss += Math.log(1 + expnyt);
				float kappa = -y * expnyt / (1+expnyt);
				
				// System.out.printf("i:%3d, y:%.1f, t:%.3f, expynt:%.3f, kappa:%.3f\n", i, y, t, expnyt, kappa);
				
				wTx(tr, i, r, model, kappa, param.eta, param.lambda, true);
			}
			tr_loss /= tr.l;
			System.out.printf("iter: %2d, tr_loss: %.5f", iter+1, tr_loss);
			
			if(va != null && va.l != 0) {
				double va_loss = 0.;
				for(int i=0; i<va.l; i++) {
					float y = va.Y[i];
					float r = R_va[i];
					float t = wTx(va, i, r, model, 0.f, 0.f, 0.f, false);
					float expnyt = (float) Math.exp(-y * t);
					va_loss += Math.log(1 + expnyt);
				}
				va_loss /= va.l;
				System.out.printf(", va_loss: %.5f", va_loss);
			}
			
			System.out.println();
		}
		
		return model;
	}
	
	public static void test(FFMModel model, FFMProblem va, FFMParameter param,
			int testBufferSize, int printInterval) {
		float[] R_va = normalize(va, param.normalization);
		LogLossEvalutor evalutor = new LogLossEvalutor(testBufferSize);
		double total_loss = 0.0;
		for(int i=0; i<va.l; i++) {
			float y = va.Y[i];
			float r = R_va[i];
			float t = wTx(va, i, r, model, 0.f, 0.f, 0.f, false);
			double expnyt = Math.exp(-y * t);
			double loss = Math.log(1 + expnyt);
			total_loss += loss;
			evalutor.addLogLoss(loss);
			if((i+1) % printInterval == 0) {
				System.out.printf("%d, %f\n", (i+1)/printInterval, evalutor.getAverageLogLoss());
			}
		}
		System.out.printf("%f\n", total_loss/va.l);
	}
	
	public static void main(String[] args) throws IOException {
//		if(args.length != 8) {
//			System.out.println("java -jar ffm.jar <eta> <lambda> <n_iters> "
//					+ "<k> <normal> <random> <train_file> <va_file>");
//			System.out.println("for example:\n"
//					+ "java -jar ffm.jar 0.1 0.01 15 4 true false tr_ va_");
//		}
		String root = "J:/Workspace/FFMInput/data/";
		// args[6] is the path of the train_file
		// tr contains the parameters and training data
		
		
		System.out.println("========== Reading the training file ========");
		FFMProblem tr = FFMProblem.readFFMProblem(root + "train_input_5.csv"); 
		// args[7] is the path of the test_file
		// va contains the parameters and test data
		System.out.println();
		System.out.println("========== Reading the test file ========");
		FFMProblem va = FFMProblem.readFFMProblem(root + "test_input_5.csv"); 
		
		System.out.println();
		System.out.println("========== Done input reading ========");
		System.out.println();
		/**
		 *  eta: used for learning rate
			lambda: used for L2 regularization
			iter: max iterations
			factor: latent factor num
			norm: instance wise normalization
			rand: use random instance order when training
			
			trset: train set
			vaset: validation set
		    
		    norm and rand only affect training speed.
			best eta is about 0.1, bigger eta hurt validation logloss, smaller eta get slow convergence.
		 */
		
		FFMParameter param = FFMParameter.defaultParameter();
		param.eta = (float) 0.01; // Learning rate
		param.lambda = (float) 1e-3; //if lambda is too large, the model is not able to achieve a good performance. With a small lambda, the model gets better results, but it easily overfits the data.
		param.n_iters = 5; // The number of iterations (manually assigned)
		param.k = 32;  // Vector dimensionality
		param.normalization = false; // Whether normalization should be done
		param.random = true;
		// cw87
		param.C = (float) 0.1;
		param.hrchyReg = true;
		
		
		System.out.println("========== Training FFM ========");
		FFMModel.train(tr, va, param);
		System.out.println("========== Done Training ========");
		System.out.println();
	}	
	

}
