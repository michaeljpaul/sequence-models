import java.io.BufferedReader;
import java.io.FileReader;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.HashMap;
import java.util.Random;
import org.apache.commons.math.special.Gamma;

public class BlockHMM extends TopicModel {

	public HashMap<Integer,Integer> docMap;
	public HashMap<String,Integer> wordMap;
	public HashMap<Integer,String> wordMapInv;

	public String[] docsID;
        public int[] docsPrev;
        public int[][] docsNext;
	public int[][] docs;
	public int[] docsZ;
	public int[][] docsX;

	public int[][] nTZ;
	public int[] nT;
	public int[][] nZW;
	public int[] nZ;
	public int[] nBW;
	public int nB;
	public int[] nX;

	public int D;
	public int W;
	public int Z;
	
	public double omega;
	public double[] alpha;
	public double gamma0;
	public double gamma1;

	public 	Random r = new Random();

	public BlockHMM(int z, double g0, double g1) {
		gamma0 = g0;
		gamma1 = g1;
		Z = z;

		alpha = new double[Z];
	        for (int i = 0; i < Z; i++) {
			alpha[i] = 0.1; // initial value
		}
	}
	
	public void initialize() {
		System.out.println("Initializing...");

		omega = 0.01; // initial value

		docsZ = new int[D];
		docsX = new int[D][];

		nTZ = new int[Z+1][Z];
		nT = new int[Z+1];
		nZW = new int[Z][W];
		nZ = new int[Z];
		nBW = new int[W];
		nB = 0;
		nX = new int[2];
		
		for (int d = 0; d < D; d++) {
			int z = r.nextInt(Z);		// select random z value in {0...Z-1}
			docsZ[d] = z;

			docsX[d] = new int[docs[d].length];
			
			for (int n = 0; n < docs[d].length; n++) {
				int w = docs[d][n];

				//int x = r.nextInt(2);		// select x uniformly
				int x = 0;
				double u = r.nextDouble();		// select random x value in {0,1}
				u *= (double)(gamma0+gamma1);		// from distribution given by prior
				if (u > gamma0) x = 1;
				//x = 1;
				docsX[d][n] = x;
				
				// update counts
				
				nX[x] += 1;
				
				if (x == 0) {
					nBW[w] += 1;
					nB += 1;
				}
				else {
					nZW[z][w] += 1;	
					nZ[z] += 1;				
				}
			}
		}
		for (int d = 0; d < D; d++) {
			int z = docsZ[d];
			int zP = docsPrev[d] == -1 ? Z : docsZ[docsPrev[d]];

			nTZ[zP][z] += 1;
			nT[zP] += 1;
		}
	}

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
		for (int d = 0; d < D; d++) {
			for (int n = 0; n < docs[d].length; n++) {
				sample(d, n);
			}
			sampleD(d);
		}

		// update hyperparams every 10 iterations
		if (iter > 100 && iter % 10 == 0) {
			// update omega
			updateOmega();

			// update alpha (Tom Minka 03 method)
			double[] alphaNew = new double[Z];
			double alphaNorm = 0;
			for (int z = 0; z < Z; z++) alphaNorm += alpha[z];

			double denom = 0;
			for (int zP = 0; zP < Z+1; zP++) {
				for (int z = 0; z < Z; z++) {
					alphaNew[z] += Gamma.digamma(nTZ[zP][z] + alpha[z]) - Gamma.digamma(alpha[z]);
				}
				denom += Gamma.digamma(nT[zP] + alphaNorm) - Gamma.digamma(alphaNorm);
			}

			for (int z = 0; z < Z; z++) {
				alpha[z] = alpha[z] * (alphaNew[z] / denom);
				alpha[z] += 0.1; // hack
				System.out.println("alpha_"+z+" "+alpha[z]);
			}			 
		}
	}
	
	public void sampleD(int d) {
		int topic = docsZ[d];
		int topicPrev = docsPrev[d] == -1 ? Z : docsZ[docsPrev[d]];

		// decrement counts

		nTZ[topicPrev][topic] -= 1;
		nT[topicPrev] -= 1;

		for (int e = 0; e < docsNext[d].length; e++) {
			nTZ[topic][docsZ[docsNext[d][e]]] -= 1;
			nT[topic] -= 1;
		}

		for (int n = 0; n < docs[d].length; n++) {
			int w = docs[d][n];
			int level = docsX[d][n];
		
			if (level == 1) {
				nZW[topic][w] -= 1;
				nZ[topic] -= 1;
			}	
		}	

		double[] logp = new double[Z+1];
	
		double alphaNorm = 0;
		for (int z = 0; z < Z; z++) alphaNorm += alpha[z];
		double omegaNorm = W * omega;

		// word probabilties
		for (int n = 0; n < docs[d].length; n++) {
			int w = docs[d][n];
			int level = docsX[d][n];

			if (level == 1) {
				for (int z = 0; z < Z; z++) {
					double prob = (nZW[z][w] + omega) / (nZ[z] + omegaNorm);
					logp[z] += Math.log(prob);
				}
			}		
		}
		// transition probabilities
		for (int z = 0; z < Z; z++) {
			double prob = (nTZ[topicPrev][z] + alpha[z]) / (nT[topicPrev] + alphaNorm);
			logp[z] += Math.log(prob);

			for (int e = 0; e < docsNext[d].length; e++) {
				int topicNext = docsZ[docsNext[d][e]];

				prob = (nTZ[z][topicNext] + alpha[topicNext]) / (nT[z] + alphaNorm);
				logp[z] += Math.log(prob);
			}
		}

		// sampled from unnormalized ratios
		double[] p = new double[Z];
		p[0] = 1.0;
		double pTotal = p[0];
		for (int z = 1; z < Z; z++) { //Z+1 if we include bg
			p[z] = p[z-1] * Math.exp(logp[z] - logp[z-1]);
			pTotal += p[z];
		}

		double u = r.nextDouble() * pTotal;
		
		double v = 0.0;
		for (int z = 0; z < Z; z++) { 
			v += p[z];
			
			if (v > u) {
				topic = z;
				break;
			}
		}

		// increment counts

		nTZ[topicPrev][topic] += 1;
		nT[topicPrev] += 1;

		for (int e = 0; e < docsNext[d].length; e++) {
			nTZ[topic][docsZ[docsNext[d][e]]] += 1;
			nT[topic] += 1;
		}

		for (int n = 0; n < docs[d].length; n++) {
			int w = docs[d][n];
			int level = docsX[d][n];
		
			if (level == 1) {
				nZW[topic][w] += 1;
				nZ[topic] += 1;
			}	
		}	
	
		docsZ[d] = topic;	
	}

	public void sample(int d, int n) {
		int w = docs[d][n];
		int topic = docsZ[d];
		int level = docsX[d][n];
		
		// decrement counts

		nX[level] -= 1;

	
		if (level == 0) {
			nBW[w] -= 1;
			nB -= 1;
		} else {
			nZW[topic][w] -= 1;
			nZ[topic] -= 1;
		}

		double omegaNorm = W * omega;

		// sample new value for level
		
		double p0 = (nX[0] + gamma0) *	// background
			(nBW[w] + omega) / (nB + omegaNorm);
		double p1 = (nX[1] + gamma1) *	// topic
			(nZW[topic][w] + omega) / (nZ[topic] + omegaNorm);

			
		double pTotal = p0 + p1;
		double u = r.nextDouble() * pTotal;
		
		if (u > p0) level = 1;
		else level = 0;

		// increment counts

		nX[level] += 1;

		if (level == 0) {
			nBW[w] += 1;
			nB += 1;
		} else {
			nZW[topic][w] += 1;
			nZ[topic] += 1;
		}
		
		// set new assignments

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
			bw.write(d+" "+docsPrev[d]+" "+docsID[d]+" "+docsZ[d]+" ");

			for (int n = 0; n < docs[d].length; n++) {
				String word = wordMapInv.get(docs[d][n]);
				bw.write(word+":"+docsX[d][n]+" ");
			}
			bw.newLine();
		}
		
		bw.close();
		fw.close();

		fw = new FileWriter(filename+".alpha");
		bw = new BufferedWriter(fw); 		

		for (int z = 0; z < Z; z++) {
			bw.write(alpha[z]+" ");
		}
		
		bw.close();
		fw.close();

		fw = new FileWriter(filename+".omega");
		bw = new BufferedWriter(fw); 		

		bw.write(""+omega);
		
		bw.close();
		fw.close();

		// estimate transition matrix
		double alphaNorm = 0;
		for (int z = 0; z < Z; z++) alphaNorm += alpha[z];

		fw = new FileWriter(filename+".pi");
		bw = new BufferedWriter(fw); 		

		for (int i = 0; i < Z; i++) {
			for (int j = 0; j < Z; j++) {
				double prob = (nTZ[i][j] + alpha[j]) / (nT[i] + alphaNorm);
				bw.write(prob+" ");
			}
			bw.newLine();
		}
		
		bw.close();
		fw.close();
	}


	/*
    public static double digamma(double x)
    {
	double r = 0.0;

	while (x <= 5.0) {
	    r -= 1.0 / x;
	    x += 1.0;
	}

	double f = 1.0 / (x * x);
	double t = f * (-1 / 12.0 + f * (1 / 120.0 + f * (-1 / 252.0 + f * (1 / 240.0 + f * (-1 / 132.0 + f * (691 / 32760.0 + f * (-1 / 12.0 
													 + f * 3617.0 / 8160.0)))))));
	return r + Math.log(x) - 0.5 / x + t;
    }
	*/
}
