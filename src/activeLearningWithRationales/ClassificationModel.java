package activeLearningWithRationales;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ThreadLocalRandom;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.regex.Pattern;

import cc.mallet.classify.Classification;
import cc.mallet.classify.NaiveBayesTrainer;
import cc.mallet.pipe.CharSequence2TokenSequence;
import cc.mallet.pipe.FeatureSequence2FeatureVector;
import cc.mallet.pipe.Input2CharSequence;
import cc.mallet.pipe.Pipe;
import cc.mallet.pipe.SerialPipes;
import cc.mallet.pipe.Target2Label;
import cc.mallet.pipe.TokenSequence2FeatureSequence;
import cc.mallet.pipe.TokenSequenceLowercase;
import cc.mallet.pipe.TokenSequenceRemoveStopwords;
import cc.mallet.pipe.iterator.CsvIterator;
import cc.mallet.types.Alphabet;
import cc.mallet.types.FeatureCounter;
import cc.mallet.types.FeatureVector;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;
import cc.mallet.types.Labeling;
import cc.mallet.types.Multinomial;

public class ClassificationModel {
	private static final Logger LOGGER = Logger.getLogger(ClassificationModel.class.getName());
	private static final String NEGATIVE_WORDS_TXT = "negative_words.txt";
	private static final String POSITIVE_WORDS_TXT = "positive_words.txt";
	public static final String NEGATIVE = "NEGATIVE";
	public static final String POSITIVE = "POSITIVE";
	static int budget = 100;
	static int bootstrap = 10;
	static double rFactor = 1;
	static double oFactor = 0.1;
	static NaiveBayesTrainer trainer = new NaiveBayesTrainer();

	static Pipe pipe;

	public ClassificationModel() {
		pipe = buildPipe();
	}

	// Reference: {@link: http://mallet.cs.umass.edu/import-devel.php}
	@SuppressWarnings({ "rawtypes", "unchecked" })
	public Pipe buildPipe() {
		ArrayList pipeList = new ArrayList();

		pipeList.add(new Input2CharSequence("UTF-8"));

		Pattern tokenPattern = Pattern.compile("[\\p{L}\\p{N}_]+");

		pipeList.add(new CharSequence2TokenSequence(tokenPattern));

		pipeList.add(new TokenSequenceLowercase());

		pipeList.add(new TokenSequenceRemoveStopwords(false, false));

		pipeList.add(new TokenSequence2FeatureSequence());

		pipeList.add(new Target2Label());

		pipeList.add(new FeatureSequence2FeatureVector());

		// pipeList.add(new PrintInputAndTarget());

		return new SerialPipes(pipeList);
	}

	public InstanceList readDirectory(String filePath) {
		CsvIterator trainReader = null;
		try {
			trainReader = new CsvIterator(new FileReader(filePath), "(\\w+)\\s+(\\w+)\\s+(.*)", 3, 2, 1);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}

		InstanceList instances = new InstanceList(pipe);
		instances.addThruPipe(trainReader);

		return instances;
	}

	// Reference for sentiment words lists: {@link:
	// https://www.cs.uic.edu/~liub/publications/www05-p536.pdf}
	public static void main(String[] args) throws IOException {

		if (args.length < 2) {
			LOGGER.log(Level.SEVERE,
					"Not enough arguments passed; Should run the jar as \"java -jar run_model.jar train_file_path test_file_path\"");
		}
		ClassificationModel importer = new ClassificationModel();
		LOGGER.info("loading train and test data");
		InstanceList instances = importer.readDirectory(args[0]);
		InstanceList testInstances = importer.readDirectory(args[1]);

		LOGGER.info("Setting tf-idf values");
		setTfIdf(instances);
		setTfIdf(testInstances);

		Alphabet completeAlphabet = instances.getAlphabet();
		Alphabet targetAlphabet = instances.getTargetAlphabet();

		// ClassLoader classLoader = ClassificationModel.class.getClassLoader();
		Map<Integer, String> positiveWordsMap = getSentimentWordList(completeAlphabet, POSITIVE_WORDS_TXT);
		Map<Integer, String> negativeWordsMap = getSentimentWordList(completeAlphabet, NEGATIVE_WORDS_TXT);

		// rFactor
		setRFactor(instances, positiveWordsMap, negativeWordsMap);
		setRFactor(testInstances, positiveWordsMap, negativeWordsMap);

		LOGGER.info("Running initial train");
		// train bootstrap
		InstanceList trainingBootstrap = new InstanceList(completeAlphabet, targetAlphabet);
		int posTrainingSamples = 0;
		int negTrainingSamples = 0;
		List<Integer> randomIndices = new ArrayList<>();
		for (int i = 0; i < 50; i++) {
			randomIndices.add(ThreadLocalRandom.current().nextInt(0, instances.size() + 1));
		}
		for (int randomInstanceIndex : randomIndices) {
			Instance currInstance = null;
			if (randomInstanceIndex < instances.size()) {
				currInstance = instances.get(randomInstanceIndex);
			} else {
				currInstance = instances.get(instances.size() - 1);
			}
			String curr_target = (String) currInstance.getTarget().toString();
			if (curr_target.equalsIgnoreCase(POSITIVE) && posTrainingSamples < bootstrap / 2) {
				instances.remove(currInstance);
				trainingBootstrap.add(currInstance);
				posTrainingSamples++;
			} else if (curr_target.equalsIgnoreCase(NEGATIVE) && negTrainingSamples < bootstrap / 2) {
				instances.remove(currInstance);
				trainingBootstrap.add(currInstance);
				negTrainingSamples++;
			}
		}

		Multinomial.Estimator featureEstimator = new Multinomial.LaplaceEstimator();
		Multinomial.Estimator priorEstimator = new Multinomial.LaplaceEstimator();
		trainer.setFeatureMultinomialEstimator(featureEstimator);
		trainer.setPriorMultinomialEstimator(priorEstimator);
		trainer.train(trainingBootstrap);

		LOGGER.info("Accuracy on test data with the 10 initial training samples is : "
				+ trainer.getClassifier().getAccuracy(testInstances));

		int initialBudget = budget;
		LOGGER.info("Training using a Budget of " + budget);
		while (budget > 0) {
			ArrayList<Classification> classify_rest = run_classifier_on_rest(instances);

			classify_rest = new ArrayList<Classification>(classify_rest.subList(0, 20));

			InstanceList uncertainSampleList = new InstanceList(completeAlphabet, targetAlphabet);
			for (Classification each_classification : classify_rest) {
				Boolean containsPos = Boolean.FALSE;
				Boolean containsNeg = Boolean.FALSE;
				Instance currInstance = each_classification.getInstance();
				FeatureVector currData = (FeatureVector) currInstance.getData();
				int[] indicesList = currData.getIndices();
				for (int eachIndex : indicesList) {
					if (positiveWordsMap.containsKey(eachIndex)) {
						containsPos = Boolean.TRUE;
					} else if (negativeWordsMap.containsKey(eachIndex)) {
						containsNeg = Boolean.TRUE;
					}
				}
				if (containsPos && containsNeg) {
					uncertainSampleList.add(currInstance);
					instances.remove(currInstance);
				}
				if (uncertainSampleList.size() == 5) {
					break;
				}
			}
			while (uncertainSampleList.size() < 5) {
				LOGGER.info("Not enough pos+neg (type 3) word cases!");
				for (Classification each_classification : classify_rest) {
					Instance currInstance = each_classification.getInstance();
					if (!uncertainSampleList.contains(currInstance)) {
						uncertainSampleList.add(currInstance);
						instances.remove(currInstance);
					}
				}
			}

			uncertainSampleList.setPipe(trainingBootstrap.getPipe());
			trainer.trainIncremental(uncertainSampleList);
			budget = budget - 5;
			LOGGER.info("Accuracy on test data with " + (initialBudget - budget + 10) + " training samples is :"
					+ trainer.getClassifier().getAccuracy(testInstances));
		}
	}

	/**
	 * @param instances
	 * @param positiveWordsMap
	 * @param negativeWordsMap
	 */
	protected static void setRFactor(InstanceList instances, Map<Integer, String> positiveWordsMap,
			Map<Integer, String> negativeWordsMap) {
		for (int i = 0; i < instances.size(); i++) {
			Instance currInstance = instances.get(i);
			FeatureVector currData = (FeatureVector) currInstance.getData();
			int[] indicesList = currData.getIndices();
			String curr_target = (String) currInstance.getTarget().toString();
			if (curr_target.equalsIgnoreCase(POSITIVE)) {
				Boolean rFactorSingleWord = Boolean.FALSE;
				for (int eachIndex : indicesList) {
					double curr_value = currData.value(eachIndex);
					if (positiveWordsMap.containsKey(eachIndex) && !rFactorSingleWord) {
						currData.setValue(eachIndex, curr_value * rFactor);
						rFactorSingleWord = Boolean.TRUE;
					} else {
						currData.setValue(eachIndex, curr_value * oFactor);
					}
				}
				currInstance.unLock();
				instances.get(i).setData(currData);
			} else if (curr_target.equalsIgnoreCase(NEGATIVE)) {
				Boolean rFactorSingleWord = Boolean.FALSE;
				for (int eachIndex : indicesList) {
					double curr_value = currData.value(eachIndex);
					if (negativeWordsMap.containsKey(eachIndex) && !rFactorSingleWord) {
						currData.setValue(eachIndex, curr_value * rFactor);
						rFactorSingleWord = Boolean.TRUE;
					} else {
						currData.setValue(eachIndex, curr_value * oFactor);
					}
				}
				currInstance.unLock();
				instances.get(i).setData(currData);
			}
		}
	}

	/**
	 * @param instances
	 */
	protected static void setTfIdf(InstanceList instances) {
		FeatureCounter counter = new FeatureCounter(instances.getDataAlphabet());

		for (int i = 0; i < instances.size(); i++) {
			Instance instance = instances.get(i);
			FeatureVector currData = (FeatureVector) instance.getData();
			int[] curr_indices = currData.getIndices();
			for (int ind : curr_indices) {
				counter.increment(ind);
			}
		}

		int numDocs = instances.size();
		for (int i = 0; i < instances.size(); i++) {
			Instance instance = instances.get(i);
			FeatureVector currData = (FeatureVector) instance.getData();
			int[] curr_indices = currData.getIndices();
			for (int ind : curr_indices) {
				double curr_value = currData.value(ind);
				currData.setValue(ind, curr_value * Math.log(numDocs / counter.get(ind)));
			}
			instances.get(i).unLock();
			instances.get(i).setData(currData);
		}
	}

	/**
	 * @param completeAlphabet
	 * @return
	 */
	protected static Map<Integer, String> getSentimentWordList(Alphabet completeAlphabet, String resourceFile) {
		Map<Integer, String> wordList = new HashMap<>();
		InputStream in = ClassificationModel.class.getResourceAsStream("/" + resourceFile);
		BufferedReader reader = new BufferedReader(new InputStreamReader(in));
		String line = null;
		try {
			while ((line = reader.readLine()) != null) {
				wordList.put(completeAlphabet.lookupIndex(line.toLowerCase()), line);
			}

		} catch (IOException e) {
			LOGGER.log(Level.SEVERE, "Error reading resource file " + resourceFile);
			e.printStackTrace();
		}
		return wordList;
	}

	private static ArrayList<Classification> run_classifier_on_rest(InstanceList instances) {
		ArrayList<Classification> classify_rest = trainer.getClassifier().classify(instances);
		Collections.sort(classify_rest, new Comparator<Classification>() {

			@Override
			public int compare(Classification o1, Classification o2) {
				Labeling label1 = o1.getLabeling();
				Labeling label2 = o2.getLabeling();
				double dist1 = Math.abs(label1.value(0) - 0.5) + Math.abs(label1.value(1) - 0.5);
				double dist2 = Math.abs(label2.value(0) - 0.5) + Math.abs(label2.value(1) - 0.5);
				if (dist1 < dist2) {
					return -1;
				} else if (dist2 < dist1) {
					return 1;
				} else {
					return 0;
				}
			}
		});
		return classify_rest;
	}

}
