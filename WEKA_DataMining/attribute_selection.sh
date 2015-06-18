java -cp "/Applications/weka-3-7-12-apple-jvm.app/Contents/Resources/Java/weka.jar" \
	 weka.filters.unsupervised.attribute.NominalToString -i Classification_dataset/MMG_data_subset.arff \
	  -o 01_NominalToString.arff -C first-last
java -cp "/Applications/weka-3-7-12-apple-jvm.app/Contents/Resources/Java/weka.jar" \
	 weka.filters.unsupervised.attribute.StringToNominal -i 01_NominalToString.arff \
	  -o 02_StringToNominal.arff -R first-last


java -cp "/Applications/weka-3-7-12-apple-jvm.app/Contents/Resources/Java/weka.jar" \
	 weka.attributeSelection.CfsSubsetEval -M -P 1 -E 1 -c 80 -x 10 -s "weka.attributeSelection.BestFirst -D 1 -N 5" -i 02_StringToNominal.arff > attributeSelection.txt

java -cp "/Applications/weka-3-7-12-apple-jvm.app/Contents/Resources/Java/weka.jar" \
	 weka.attributeSelection.CfsSubsetEval -M -P 1 -E 1 -c 30 -s "weka.attributeSelection.BestFirst -D 1 -N 5" -i 02_StringToNominal.arff > attributeSelection2.txt

 java -cp "/Applications/weka-3-7-12-apple-jvm.app/Contents/Resources/Java/weka.jar" \
	 weka.attributeSelection.CfsSubsetEval -M -P 1 -E 1 -c 14 -s "weka.attributeSelection.BestFirst -D 1 -N 5" -i 02_StringToNominal.arff > attributeSelection3.txt