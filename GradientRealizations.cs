using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Runtime.Intrinsics.X86;
using System.Text;
using System.Threading.Tasks;

namespace ConsoleApp1
{
    public abstract class AbGradientDescent
    {
        protected NeuralNetwork currentNN;
        public ILearnRegularization regularization { get; private set; }
        protected List<Layer> neuronLayer;
        protected Topology topology;
        protected IOptimizer Optimizer;
        protected List<Regularizations> regularizations;
        protected int CurrentEpoch;
        protected DelegateforWeights regulatorW;
        public AbGradientDescent(NeuralNetwork nn, IOptimizer optimizer = null)
        {
            currentNN = nn;
            neuronLayer = currentNN.neuronLayer;
            topology = currentNN.Topology;
            if (optimizer == null)
            {
                optimizer = new DefaultOptimizer();
                Optimizer = optimizer;
            }
            else
            {
                Optimizer = optimizer;
            }
            currentNN.Reg = new DefaultRegularization();
            regularization = new DefaultRegularization();
            regulatorW = (double d) => { return 0; };
            if (Optimizer.regulList != null && Optimizer.regulList.Count != 0)
            {
                Regularization regu = new Regularization();
                int IsComb = 0;
                for (int i = 0; i < Optimizer.regulList.Count; i++)
                {
                    Regularizations reg = Optimizer.regulList[i];
                    switch (reg)
                    {
                        case Regularizations.L1:
                            regulatorW = regu.L1;
                            IsComb++;
                            break;
                        case Regularizations.L2:
                            regulatorW = regu.L2;
                            IsComb++;
                            break;
                        case Regularizations.Dropout:
                            regularization = new RegDropoutLearn();
                            currentNN.Reg = new RegDropoutTest();
                            break;
                            /*case Regularizations.BatchNormalization:
                                regularization = new BatchNormalizationLearn();
                                currentNN.Reg
                                break;
                            */
                    }
                }
                if (IsComb == 2)
                {
                    regulatorW = regu.L1andL2;
                }
            }
        }
        public abstract double Backpropagation<T>(T dataset, int[] expected, int epoch);
        public void Backpropagation(double learningRate, double error)
        {
            int count = neuronLayer.Count;
            Layer currentLayer = neuronLayer[count - 1];
            Layer previousLayer = neuronLayer[count - 2];
            List<double> outputs = previousLayer.GetSignals();

            for (int j = 0; j < currentLayer.Count; j++)
            {
                Neuron currentNeuron = currentLayer[j];
                currentNeuron.Delta = Optimizer.BackPropagation(currentNeuron, outputs, error, learningRate, CurrentEpoch, regulatorW);
            }

            for (int i = neuronLayer.Count - 2; i >= 1; i--)
            {
                currentLayer = neuronLayer[i];
                Layer forwardLayer = neuronLayer[i + 1];
                previousLayer = neuronLayer[i - 1];

                outputs = previousLayer.GetSignals();
                for (int j = 0; j < currentLayer.Count; j++)
                {
                    Neuron currentNeuron = currentLayer[j];
                    double deltaSum = Optimizer.GetFromAllDelta(forwardLayer, j);
                    currentNeuron.Delta = Optimizer.BackPropagation(currentNeuron, outputs, deltaSum, learningRate, CurrentEpoch, regulatorW);
                }
            }
        }
        protected double GetErrors(double[] outputs, int expected)
        {
            return regularization.FeedForward(outputs, currentNN.neuronLayer) - expected;
        }
    }

    public class GradientDescent : AbGradientDescent
    {
        public GradientDescent(NeuralNetwork nn, IOptimizer optimizer = null) : base(nn, optimizer)
        {
        }

        public override double Backpropagation<T>(T dataset, int[] expected, int epoch)
        {
            if (dataset is double[] outputsL)
            {
                double MSE = 0;
                for (int i = 0; i < epoch; i++)
                {
                    double error = 0;
                    double MSEtemp = 0;
                    CurrentEpoch++;
                    error += GetErrors(outputsL, expected[i]);
                    MSEtemp += error * error;
                    MSEtemp = MSEtemp / outputsL.Length;
                    MSE += MSEtemp;
                    error = error / outputsL.Length;
                    Backpropagation(topology.LearningRate, error);
                }
                return MSE / epoch;
            }
            else if (dataset is double[][] outputsM)
            {
                double MSE = 0;
                for (int j = 0; j < epoch; j++)
                {
                    double error = 0;
                    double MSEtemp = 0;
                    CurrentEpoch++;
                    for (int i = 0; i < expected.Length; i++)
                    {
                        error = GetErrors(outputsM[i], expected[i]);
                        MSEtemp += error * error;
                    }
                    MSEtemp = MSEtemp / expected.Length;
                    error = error / expected.Length;
                    MSE += MSEtemp;
                    Backpropagation(topology.LearningRate, error);
                }
                return MSE / epoch;
            }
            else
            {
                return 0;
            }
        }
    }

    public class MiniBatchGD : AbGradientDescent
    {
        Random random;
        int Capacity;
        public MiniBatchGD(NeuralNetwork nn, int capacity = 2, IOptimizer optimizer = null) : base(nn, optimizer) 
        {
            if (capacity < 1)
            {
                return;
            }
            Capacity = capacity; 
            random = new Random();
        }

        public override double Backpropagation<T>(T dataset, int[] expected, int epoch)
        {
            if (dataset is double[][] outputsM)
            {
                double MSE = 0;
                for (int j = 0; j < epoch; j++)
                {
                    double error = 0;
                    double MSEtemp = 0;
                    CurrentEpoch++;
                    double learningRate = 0.1 / Math.Pow(CurrentEpoch, 0.3);
                    for (int i = 0; i < Capacity; i++)
                    {
                        int rnd = random.Next(0, expected.Length - 1);
                        error += GetErrors(outputsM[rnd], expected[rnd]);
                        MSEtemp += error * error;
                    }
                    MSEtemp = MSEtemp / outputsM.Length;
                    MSE += MSEtemp;
                    error = error / outputsM.Length;
                    Backpropagation(learningRate, error);
                }
                return MSE / epoch;
            }
            else
            {
                return 0;
            }
        }
    }

    public class StochasticGradient : AbGradientDescent
    {
        Random random;
        public StochasticGradient(NeuralNetwork nn, IOptimizer optimizer = null) : base(nn, optimizer)
        {
            random = new Random();
        }

        public override double Backpropagation<T>(T dataset, int[] expected, int epoch)
        {
            if (dataset is double[][] outputsM)
            {
                double MSE = 0;
                for (int i = 0; i < epoch; i++)
                {
                    CurrentEpoch++;
                    double learningRate = 0.1 / Math.Pow(CurrentEpoch, 0.3);
                    int rnd = random.Next(0, expected.Length - 1);
                    double error = GetErrors(outputsM[rnd], expected[rnd]);
                    double MSEtemp = error * error;
                    MSEtemp = MSEtemp / outputsM.Length;
                    MSE += MSEtemp;
                    error = error / outputsM.Length;
                    Backpropagation(learningRate, error);
                }
                return MSE / epoch;
            }
            else
            {
                return 0;
            }
        }
    }
}
