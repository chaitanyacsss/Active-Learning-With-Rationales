package activeLearningWithRationales;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.junit.Test;

import cc.mallet.types.Alphabet;
import cc.mallet.types.FeatureVector;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;
import static org.junit.Assert.assertTrue;

public class ClassificationModelTest {

	String dataFile = "resources/mallet-sample.msv";
	private static final String NEGATIVE_WORDS_TXT = "negative_words.txt";
	private static final String POSITIVE_WORDS_TXT = "positive_words.txt";
	double delta = Math.pow(10, -7);

	@Test
	public void testReadDirectory() {
		ClassificationModel cm = new ClassificationModel();
		InstanceList instances = cm.readDirectory(dataFile);
		assertNotNull(instances);
		assertEquals(50, instances.size());
	}

	@Test
	public void testSetTfIdf() {
		ClassificationModel cm = new ClassificationModel();
		InstanceList instances = cm.readDirectory(dataFile);
		Alphabet alphabet = instances.getAlphabet();
		ClassificationModel.setTfIdf(instances);

		// Eddie and Murphy are single occurrence terms. Such non-repeating
		// words should have 1*log(50) (base e) as value
		int index1 = alphabet.lookupIndex("eddie");
		int index2 = alphabet.lookupIndex("murphy");

		// film/movie should have 0 as value since it occurs 70 times (> no. of
		// instances)
		int index3 = alphabet.lookupIndex("film");
		int index4 = alphabet.lookupIndex("movie");

		for (int i = 0; i < instances.size(); i++) {
			Instance currentInstance = instances.get(i);
			if (((FeatureVector) currentInstance.getData()).contains("film")) {
				assertEquals(0, ((FeatureVector) currentInstance.getData()).getValues()[index3], delta);
			}
			if (((FeatureVector) currentInstance.getData()).contains("movie")) {
				assertEquals(0, ((FeatureVector) currentInstance.getData()).getValues()[index4], delta);
			}

		}
		assertEquals(Math.log(instances.size()), ((FeatureVector) instances.get(0).getData()).getValues()[index1],
				delta);
		assertEquals(Math.log(instances.size()), ((FeatureVector) instances.get(0).getData()).getValues()[index2],
				delta);

	}

	@Test
	public void testGetSentimentWordList() {
		ClassificationModel cm = new ClassificationModel();
		InstanceList instances = cm.readDirectory(dataFile);
		Alphabet alphabet = instances.getAlphabet();
		Map<Integer, String> positiveWordsMap = ClassificationModel.getSentimentWordList(alphabet, POSITIVE_WORDS_TXT);
		Map<Integer, String> negativeWordsMap = ClassificationModel.getSentimentWordList(alphabet, NEGATIVE_WORDS_TXT);

		assertTrue(positiveWordsMap.containsValue("great"));
		assertTrue(positiveWordsMap.containsValue("amazing"));
		assertTrue(positiveWordsMap.containsValue("fantastic"));
		assertEquals(2006, positiveWordsMap.size());

		assertTrue(negativeWordsMap.containsValue("bad"));
		assertTrue(negativeWordsMap.containsValue("terrible"));
		assertTrue(negativeWordsMap.containsValue("sucks"));
		assertEquals(4783, negativeWordsMap.size());
	}

	@Test
	public void testSetRFactor() {
		ClassificationModel cm = new ClassificationModel();
		InstanceList instances = cm.readDirectory(dataFile);
		Alphabet alphabet = instances.getAlphabet();
		ClassificationModel.setTfIdf(instances);

		Map<Integer, String> positiveWordsMap = ClassificationModel.getSentimentWordList(alphabet, POSITIVE_WORDS_TXT);
		Map<Integer, String> negativeWordsMap = ClassificationModel.getSentimentWordList(alphabet, NEGATIVE_WORDS_TXT);

		List<Double[]> preModifiedValuesList = new ArrayList<>();
		for (int i = 0; i < instances.size(); i++) {
			Instance currentInstance = instances.get(i);
			double[] valuesFromInstances = ((FeatureVector) currentInstance.getData()).getValues();
			Double[] valuesToSet = new Double[valuesFromInstances.length];
			for (int j = 0; j < valuesFromInstances.length; j++) {
				Double currValue = valuesFromInstances[j];
				valuesToSet[j] = currValue;
			}
			preModifiedValuesList.add(valuesToSet);
		}

		ClassificationModel.setRFactor(instances, positiveWordsMap, negativeWordsMap);

		for (int i = 0; i < instances.size(); i++) {
			Instance currentInstance = instances.get(i);
			Double[] oldValues = preModifiedValuesList.get(i);
			double[] newValues = ((FeatureVector) currentInstance.getData()).getValues();

			List<Double> factorOfChange = new ArrayList<>();
			int numberOfChanges = 0;
			for (int j = 0; j < newValues.length; j++) {
				if ((oldValues[j].compareTo(newValues[j])) != 0) {
					factorOfChange.add(newValues[j] / oldValues[j]);
					numberOfChanges++;
				} else if (oldValues[j].equals(0.0)) {
					numberOfChanges++;
				}
			}

			assertEquals(oldValues.length - 1, numberOfChanges);
			for (Double eachFactor : factorOfChange) {
				assertEquals(0.1, eachFactor, delta);
			}
			assertTrue(factorOfChange.contains(0.1));

		}

	}

}
