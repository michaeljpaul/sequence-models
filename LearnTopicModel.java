import java.util.HashMap;

public class LearnTopicModel {

	public static HashMap<String,String> arguments;
	
	public static void main(String[] args) throws Exception {
		arguments = new HashMap<String,String>();
		
		for (int i = 0; i < args.length; i += 2) {
			arguments.put(args[i], args[i+1]);
		}

		String model = arguments.get("-model");
		String filename = arguments.get("-input");
		
		if (model == null) {
			System.out.println("No model specified.");
			return;
		}
		
		if (filename == null) {
			System.out.println("No input file given.");
			return;
		}
		
		TopicModel topicModel = null;

		if (model.equals("m4") || model.equals("M4")) {
			if (!arguments.containsKey("-Z")) {
				System.out.println("Must specify number of topics using -Z");
				return;
			}
			int Z = Integer.parseInt(arguments.get("-Z"));
			
			double gamma0 = 1.0;
			double gamma1 = 1.0;
			double sigma2 = 10.0;
			double eta = 1.0;

			int rightContext = 0;

			if (arguments.containsKey("-gamma0")) 
				gamma0 = Double.parseDouble(arguments.get("-gamma0"));
			if (arguments.containsKey("-gamma1")) 
				gamma1 = Double.parseDouble(arguments.get("-gamma1"));
			if (arguments.containsKey("-sigma2")) 
				sigma2 = Double.parseDouble(arguments.get("-sigma2"));
			if (arguments.containsKey("-eta")) 
				eta = Double.parseDouble(arguments.get("-eta"));
			if (arguments.containsKey("-rightContext")) 
				rightContext = Integer.parseInt(arguments.get("-rightContext"));
			
			topicModel = new M4(Z, gamma0, gamma1, sigma2, eta, rightContext);
		}
		else if (model.equals("hmm") || model.equals("HMM")) {
			if (!arguments.containsKey("-Z")) {
				System.out.println("Must specify number of topics using -Z");
				return;
			}
			
			int Z = Integer.parseInt(arguments.get("-Z"));
			
			double gamma0 = 1.0;
			double gamma1 = 1.0;

			if (arguments.containsKey("-gamma0")) 
				gamma0 = Double.parseDouble(arguments.get("-gamma0"));
			if (arguments.containsKey("-gamma1")) 
				gamma1 = Double.parseDouble(arguments.get("-gamma1"));
			
			topicModel = new BlockHMM(Z, gamma0, gamma1);
		}
		else {
			System.out.println("Invalid model specification. Options: m4 | hmm");
			return;
		}
		
		int iters = 1000;
		if (arguments.containsKey("-iters")) 
			iters = Integer.parseInt(arguments.get("-iters"));
		
		topicModel.run(iters, filename);
	}

}
