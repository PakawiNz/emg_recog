public boolean batchFinished() throws Exception {
	if (getInputFormat() == null)
		throw new IllegalStateException("No input instance format defined");
	if (m_MinArray == null) {
		Instances input = getInputFormat();
		// Compute minimums and maximums
		m_MinArray = new double[input.numAttributes()];
		m_MaxArray = new double[input.numAttributes()];
		for (int i = 0; i < input.numAttributes(); i++)
			m_MinArray[i] = Double.NaN;

		for (int j = 0; j < input.numInstances(); j++) {
			double[] value = input.instance(j).toDoubleArray();
			for (int i = 0; i < input.numAttributes(); i++) {
				if (input.attribute(i).isNumeric() && (input.classIndex() != i)) {
					if (!Instance.isMissingValue(value[i])) {
						if (Double.isNaN(m_MinArray[i])) {
							m_MinArray[i] = m_MaxArray[i] = value[i];
						} else {
							if (value[i] < m_MinArray[i])
								m_MinArray[i] = value[i];
							if (value[i] > m_MaxArray[i])
								m_MaxArray[i] = value[i];
						}
					}
				}
			}
		}
		// Convert pending input instances
		for (int i = 0; i < input.numInstances(); i++)
			convertInstance(input.instance(i));
	}
	// Free memory
	flushInput();
	m_NewBatch = true;
	return (numPendingOutput() != 0);
}

protected void convertInstance(Instance instance) throws Exception {
	Instance inst = null;
	if (instance instanceof SparseInstance) {
		double[] newVals = new double[instance.numAttributes()];
		int[] newIndices = new int[instance.numAttributes()];
		double[] vals = instance.toDoubleArray();
		int ind = 0;
		for (int j = 0; j < instance.numAttributes(); j++) {
			double value;
			if (instance.attribute(j).isNumeric() && (!Instance.isMissingValue(vals[j])) && (getInputFormat().classIndex() != j)) {
				if (Double.isNaN(m_MinArray[j]) || (m_MaxArray[j] == m_MinArray[j])) {
					value = 0;
				} else {
					value = (vals[j] - m_MinArray[j]) / (m_MaxArray[j] - m_MinArray[j]) * m_Scale + m_Translation;
					if (Double.isNaN(value)) {
						throw new Exception("A NaN value was generated while normalizing " + instance.attribute(j).name());
					}
				}
				if (value != 0.0) {
					newVals[ind] = value;
					newIndices[ind] = j;
					ind++;
				}
			} else {
				value = vals[j];
				if (value != 0.0) {
					newVals[ind] = value;
					newIndices[ind] = j;
					ind++;
				}
			}
		}
		double[] tempVals = new double[ind];
		int[] tempInd = new int[ind];
		System.arraycopy(newVals, 0, tempVals, 0, ind);
		System.arraycopy(newIndices, 0, tempInd, 0, ind);
		inst = new SparseInstance(instance.weight(), tempVals, tempInd,instance.numAttributes());
	} else {
		double[] vals = instance.toDoubleArray();
		for (int j = 0; j < getInputFormat().numAttributes(); j++) {
			if (instance.attribute(j).isNumeric() && (!Instance.isMissingValue(vals[j])) && (getInputFormat().classIndex() != j)) {
				if (Double.isNaN(m_MinArray[j]) || (m_MaxArray[j] == m_MinArray[j])) {
					vals[j] = 0;
				} else {
					vals[j] = (vals[j] - m_MinArray[j]) / (m_MaxArray[j] - m_MinArray[j]) * m_Scale + m_Translation;
					if (Double.isNaN(vals[j])) {
						throw new Exception("A NaN value was generated " + "while normalizing " + instance.attribute(j).name());
					}
				}
			}
		}
		inst = new Instance(instance.weight(), vals);
	}
	inst.setDataset(instance.dataset());
	push(inst);
}

// Multilayer Perceptron Normalization

private Instances setClassType(Instances inst) throws Exception {
	if (inst != null) {
		// x bounds
		double min=Double.POSITIVE_INFINITY;
		double max=Double.NEGATIVE_INFINITY;
		double value;
		m_attributeRanges = new double[inst.numAttributes()];
		m_attributeBases = new double[inst.numAttributes()];
		for (int noa = 0; noa < inst.numAttributes(); noa++) {
			min = Double.POSITIVE_INFINITY;
			max = Double.NEGATIVE_INFINITY;
			for (int i=0; i < inst.numInstances();i++) {
				if (!inst.instance(i).isMissing(noa)) {
					value = inst.instance(i).value(noa);
					if (value < min) {
						min = value;
					}
					if (value > max) {
						max = value;
					}
			  	}
			}
			
			m_attributeRanges[noa] = (max - min) / 2;
			m_attributeBases[noa] = (max + min) / 2;
			if (noa != inst.classIndex() && m_normalizeAttributes) {
				for (int i = 0; i < inst.numInstances(); i++) {
					if (m_attributeRanges[noa] != 0) {
						inst.instance(i).setValue(noa, (inst.instance(i).value(noa) - m_attributeBases[noa]) / m_attributeRanges[noa]);
					} else {
						inst.instance(i).setValue(noa, inst.instance(i).value(noa) - m_attributeBases[noa]);
					}
				}
			}
		}
		if (inst.classAttribute().isNumeric()) {
			m_numeric = true;
		} else {
			m_numeric = false;
		}
	}
	return inst;
}