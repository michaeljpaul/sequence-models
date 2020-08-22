import java.io.BufferedReader;
import java.io.FileReader;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.HashMap;
import java.util.Random;
import org.apache.commons.math.special.Gamma;

public class M4 extends TopicModel {

	public HashMap<Integer,Integer> docMap;
	public HashMap<String,Integer> wordMap;
	public HashMap<Integer,String> wordMapInv;

	public String[] docsID;
        public int[] docsPrev;
        public int[][] docsNext;
	public int[][] docs;
	public int[][] docsZ;
	public int[][] docsX;

	public int[][] nDZ;
	public int[] nD;
	public int[][] nZW;
	public int[] nZ;
	public int[] nBW;
	public int nB;
	public int[] nX;

	public int D;
	public int W;
	public int Z;
	
	public double omega;
	public double[][] theta;
	public double[] thetaSum;
        public double[][] lambda;
	public double gamma0;
	public double gamma1;
	public double sigma2;
	public double stepSize;
	public double eta;

	private int rightContext;

	private Random r = new Random();

	public M4(int z, double g0, double g1, double s, double e, int rc) {
		Z = z;
		gamma0 = g0;
		gamma1 = g1;
		sigma2 = s;
		eta = e;
		rightContext = rc;
	}
	
	public void initialize() {
		System.out.println("Initializing...");

		int rc = rightContext;
		if (rc < 0) { // never sample right context
			rc = iters + 1;
		} else if (rc == 0) { // always sample right context
			rc = 1;
		}
		rightContext = rc;

		docsZ = new int[D][];
		docsX = new int[D][];

		theta = new double[D][Z];
		thetaSum = new double[D];
		lambda = new double[Z][Z+2];
		omega = 0.01; // initial value

		nDZ = new int[D][Z+1];
		nD = new int[D];
		nZW = new int[Z][W];
		nZ = new int[Z];
		nBW = new int[W];
		nB = 0;
		nX = new int[2];
		
		for (int d = 0; d < D; d++) {
			docsZ[d] = new int[docs[d].length];
			docsX[d] = new int[docs[d].length];
			
			for (int n = 0; n < docs[d].length; n++) {
				int w = docs[d][n];

				int z = r.nextInt(Z);		// select random z value in {0...Z-1}
				docsZ[d][n] = z;

				//int x = r.nextInt(2);		// select x uniformly
				int x = 0;
				double u = r.nextDouble();		// select random x value in {0,1}
				u *= (double)(gamma0+gamma1);		// from distribution given by prior
				if (u > gamma0) x = 1;
				docsX[d][n] = x;
				
				// update counts
				
				nX[x] += 1;
				
				if (x == 0) {
					nBW[w] += 1;
					nB += 1;
				}
				else {
					nDZ[d][z] += 1;
					nD[d] += 1;
					nZW[z][w] += 1;	
					nZ[z] += 1;				
				}
			}
		}

		updateTheta();
	}


	public double dotProduct(int[] a, double[] b) {
		// first block in document ("start" feature is on)
		if (a[Z] == 1) return b[Z];
	    	
		double x = 0;
		for (int i = 0; i <= Z; i++) {
			x += a[i]*b[i];
		}
		return x;
	}

	public void updateLambda() {
		System.out.println("Updating lambda...");

		int[] nD0 = new int[Z+1]; // topic counts for null document
		nD0[Z] = 1; // "no-parent" feature is index Z

		double[][] gradient = new double[Z][Z+2];

		for (int d = 0; d < D; d++) {
			int d0 = docsPrev[d];
			int [] x = (d0 == -1) ? nD0 : nDZ[d0];

			// normalize doc counts -- add 1 to denominator to ensure nonzero
			double norm = 1.0;
			if (d0 != -1) norm = 1.0 / (nD[d0] + 1.0);

			for (int t = 0; t < Z; t++) {
				double EnDt = nD[d] * theta[d][t]; // expected counts of t

				int k = 0;
				for ( ; k <= Z; k++) {
					gradient[t][k] += norm * x[k] * (nDZ[d][t] - EnDt);
				}
				gradient[t][k] += (nDZ[d][t] - EnDt); // bias
			}
		}

		for (int t = 0; t < Z; t++) {
			for (int k = 0; k < Z+2; k++) {
				gradient[t][k] -= lambda[t][k] / sigma2;

				lambda[t][k] = lambda[t][k] + stepSize*gradient[t][k];
				if (lambda[t][k] < -30.0) lambda[t][k] = -30.0; //hack-- keep from diverging
				if (lambda[t][k] > 30.0) lambda[t][k] = 30.0;

				System.out.print(lambda[t][k]+" ");
			}
			System.out.println();
		}
        }

	// theta is the cached value of the class distribution for each document
	public void updateTheta() {
		for (int d = 0; d < D; d++) {
			updateTheta(d);
		}
	}

	public void updateTheta(int d) {
		int[] nD0 = new int[Z+1];
		nD0[Z] = 1;

		int d0 = docsPrev[d];

		// normalize doc counts -- add 1 to denominator to ensure nonzero
		double norm = 1.0;
		if (d0 != -1) norm = 1.0 / (nD[d0] + 1.0);

		int[] x = (d0 == -1) ? nD0 : nDZ[d0];

		thetaSum[d] = 0;
		double dotProd;
		for (int t = 0; t < Z; t++) {
			// normalized for first Z cells of vector; Z+1 is bias
			dotProd = norm*dotProduct(x, lambda[t]) + lambda[t][Z+1];
			theta[d][t] = Math.exp(dotProd);
			thetaSum[d] += theta[d][t];
		}

		double thetaSum2 = 0;
		for (int t = 0; t < Z; t++) {
			theta[d][t] = theta[d][t]/thetaSum[d];
			//if (new Double(theta[d][t]).isNaN()) theta[d][t] = 1.0;
		}
		thetaSum[d] = 1.0;
	}

	// omega is the word smoothing concentration parameter.
	// this is sampled.
	public void updateOmega()
	{
		double LLold = 0;
		double LLnew = 0;
		double omegaSum = omega*(double)W;

                double omegaNew = Math.exp(Math.log(omega) + r.nextGaussian());
                double omegaSumNew = omegaNew * (double)W;

		for (int z = 0; z < Z; z++) {
			LLold += Gamma.logGamma(omegaSum) - Gamma.logGamma(nZ[z] + omegaSum);
			LLnew += Gamma.logGamma(omegaSumNew) - Gamma.logGamma(nZ[z] + omegaSumNew);

			for (int w = 0; w < W; w++) {
				LLold += Gamma.logGamma(nZW[z][w] + omega) - Gamma.logGamma(omega);
				LLnew += Gamma.logGamma(nZW[z][w] + omegaNew) - Gamma.logGamma(omegaNew);
			}
		}

		double ratio = Math.exp(LLnew - LLold);

		boolean accept = false;
		if (r.nextDouble() < ratio) accept = true;
		if (omegaNew > 0.5) accept = false; // hack

		System.out.println("omega: proposed "+omegaNew);
		System.out.println(" (ratio = "+ratio);
		if (accept) {
			omega = omegaNew;
			System.out.println("Accepted");
		} else {
			System.out.println("Rejected");
		}
		System.out.println("omega: "+omega);
	}
	
	public void doSampling() {
		stepSize = eta / (1000.0 + iter);
		updateLambda();

		System.out.println("Sampling...");
		for (int d = 0; d < D; d++) {
			updateTheta(d);
			
			for (int n = 0; n < docs[d].length; n++) {
				sample(d, n);
			}
		}

		if (iter >= 100) updateOmega();
	}
	
	public void sample(int d, int n) {
		int w = docs[d][n];
		int topic = docsZ[d][n];
		int level = docsX[d][n];
		int d0 = docsPrev[d];	
	
		// decrement counts

		nX[level] -= 1;

		if (level == 0) {
			nBW[w] -= 1;
			nB -= 1;
		} else {
			nDZ[d][topic] -= 1;
			nD[d] -= 1;
			nZW[topic][w] -= 1;
			nZ[topic] -= 1;
		}

		double omegaNorm = W * omega;
		double dotProd;

		// sample new value for level
		
		double pTotal = 0.0;
		double[] logp = new double[Z+1];
		double[] p = new double[Z+1];

		// this will be p(x=0) (background)	
		
		p[Z] = (nX[0] + gamma0) *
			(nBW[w] + omega) / (nB + omegaNorm);
		logp[Z] = Math.log(p[Z]);

		double[] xL = new double[Z];
		double[] ExL = new double[Z];

		// normalize doc counts -- add 1 to denominator to ensure nonzero
		double norm = 1.0 / (nD[d] + 1);

		if (docsNext[d].length > 0 &&
			iter % rightContext == 0) { // consider next documents as if no topic was assigned here
			double ExLsum = 0;
			for (int j = 0; j < Z; j++) {
				xL[j] = norm*dotProduct(nDZ[d], lambda[j]) + lambda[j][Z+1];
				ExL[j] = Math.exp(xL[j]);
				ExLsum += ExL[j];
			}

			for (int k = 0; k < docsNext[d].length; k++) {
				int c = docsNext[d][k];

				for (int j = 0; j < Z; j++) {
					logp[Z] += nDZ[c][j] * Math.log(ExL[j] / ExLsum);
				}
			}
		}

		// sample new value for topic and level

		for (int z = 0; z < Z; z++) {
			logp[z] = (nX[1] + gamma1) * 
				(theta[d][z]) *
				(nZW[z][w] + omega) / (nZ[z] + omegaNorm);
				logp[z] = Math.log(logp[z]);
		    
			if (docsNext[d].length == 0 || iter % rightContext != 0) continue;
		   
			// consider next documents as if this topic were assigned 
			double ExLsum = 0;
			for (int j = 0; j < Z; j++) {	
				nDZ[d][z] += 1;
				ExL[j] = Math.exp(xL[j] + norm*lambda[j][z]);
				ExLsum += ExL[j];
				nDZ[d][z] -= 1;
			}
		    
			for (int k = 0; k < docsNext[d].length; k++) {
				int c = docsNext[d][k];

				for (int j = 0; j < Z; j++) {
					logp[z] += nDZ[c][j] * Math.log(ExL[j] / ExLsum);
				}
			}
		}

		// sample in log space
		// see e.g. http://blog.smola.org/post/987977550/log-probabilities-semirings-and-floating-point-numbers
		double maxLL = Double.NEGATIVE_INFINITY;
		for (int z = 0; z < Z+1; z++) { 
			if (logp[z] > maxLL) {
				maxLL = logp[z];
			}
		}

		double pSum = 0;
		for (int z = 0; z < Z+1; z++) {
			pSum += Math.exp(logp[z] - maxLL);
		}
		pSum = Math.log(pSum);
		pSum += maxLL;

		pTotal = 0;
		for (int z = 0; z < Z+1; z++) { 
			p[z] = Math.exp(logp[z] - pSum);
			pTotal += p[z];
		}
		if (pTotal > 1.01 || pTotal < 0.99) System.out.println("OOPS "+pTotal);

		double u = r.nextDouble() * pTotal;
		
		double v = 0.0;
		for (int z = 0; z < Z+1; z++) { 
			v += p[z];
			
			if (v > u) {
				topic = z;
				break;
			}
		}

		if (topic == Z) level = 0;
		else level = 1;

		// increment counts

		nX[level] += 1;

		if (level == 0) {
			nBW[w] += 1;
			nB += 1;
		} else {
			nDZ[d][topic] += 1;
			nD[d] += 1;
			nZW[topic][w] += 1;
			nZ[topic] += 1;
		}
		
		// set new assignments

		docsZ[d][n] = topic;
		docsX[d][n] = level;
	}

	public void readDocs(String filename) throws Exception {
		System.out.println("Reading input...");
		
		docMap = new HashMap<Integer,Integer>();
		wordMap = new HashMap<String,Integer>();
		wordMapInv = new HashMap<Integer,String>();
		
		FileReader fr = new FileReader(filename);
		BufferedReader br = new BufferedReader(fr); 

		String s;
	
		D = 0;
		int dj = 0;
		while((s = br.readLine()) != null) {
			D++;
		}

		docsID = new String[D];
		docs = new int[D][];
		docsPrev = new int[D];
		docsNext = new int[D][];
		int[] countNext = new int[D];

		fr = new FileReader(filename);
		br = new BufferedReader(fr); 
		while ((s = br.readLine()) != null) {
			String[] tokens0 = s.split("\\s+");
			String [] tokens = new String[tokens0.length-3];
			for (int n = 3; n < tokens0.length; n++) tokens[n-3] = tokens0[n];			

			int d0 = Integer.parseInt(tokens0[0]);
			if (!docMap.containsKey(d0)) {
				docMap.put(d0, docMap.size());
			}
		}
		fr.close();
		br.close();

		fr = new FileReader(filename);
		br = new BufferedReader(fr); 

		int d = 0;
		int di = 0;
		while ((s = br.readLine()) != null) {
			String[] tokens0 = s.split("\\s+");
			String [] tokens = new String[tokens0.length-3];
			for (int n = 3; n < tokens0.length; n++) tokens[n-3] = tokens0[n];			
			
			int N = tokens.length;

			int d0 = Integer.parseInt(tokens0[0]);
			d = docMap.get(d0);

			docsPrev[d] = Integer.parseInt(tokens0[1]);
			docsID[d] = tokens0[2];
			if (docsPrev[d] > -1) {
				docsPrev[d] = docMap.get(docsPrev[d]); // if not -1, convert to internal ID
				countNext[docsPrev[d]] += 1;
			}

			docs[d] = new int[N];
			
			for (int n = 0; n < N; n++) {
				String word = tokens[n];
				
				int key = wordMap.size();
				if (!wordMap.containsKey(word)) {
					wordMap.put(word, new Integer(key));
					wordMapInv.put(new Integer(key), word);
				}
				else {
					key = ((Integer) wordMap.get(word)).intValue();
				}
				
				docs[d][n] = key;
			}
		}

		for (int i = 0; i < D; i++) docsNext[i] = new int[countNext[i]];
		countNext = new int[D];
		for (int i = 0; i < D; i++) {
			int prev = docsPrev[i];
			if (prev != -1) {
				docsNext[prev][countNext[prev]] = i;
				countNext[prev]++;
			}
		}
		
		br.close();
		fr.close();
		
		W = wordMap.size();

		System.out.println(D+" documents");
		System.out.println(W+" word types");
	}

	public void writeOutput(String filename) throws Exception {
		System.out.println("Writing output...");

		FileWriter fw = new FileWriter(filename+".assign");
		BufferedWriter bw = new BufferedWriter(fw); 		

		for (int d = 0; d < D; d++) {
			bw.write(d+" ");
			bw.write(docsPrev[d]+" ");
			bw.write(docsID[d]+" ");

			for (int n = 0; n < docs[d].length; n++) {
				String word = wordMapInv.get(docs[d][n]);
				bw.write(word+":"+docsZ[d][n]+":"+docsX[d][n]+" ");
			}
			bw.newLine();
		}
		
		bw.close();
		fw.close();

		fw = new FileWriter(filename+".lambda");
		bw = new BufferedWriter(fw); 		

		for (int t = 0; t < Z; t++) {
			for (int k = 0; k < Z+2; k++) {
				bw.write(lambda[t][k]+" ");
			}
			bw.newLine();
		}
		
		bw.close();
		fw.close();

		fw = new FileWriter(filename+".omega");
		bw = new BufferedWriter(fw); 		

		bw.write(""+omega);
		
		bw.close();
		fw.close();
	}

}
