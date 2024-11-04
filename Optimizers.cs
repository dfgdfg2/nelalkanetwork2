using ConsoleApp1;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConsoleApp1
{
    public abstract class IOptimizer
    {
        public List<Regularizations> regulList { get; set; }
        public IOptimizer() 
        { 
        }
        public IOptimizer(List<Regularizations> regularizations)
        {
            regulList = regularizations;
            if (regulList.Count != regulList.Distinct().Count())
            {
                throw new ArgumentException("Can be only 1 unical modifier");
            }
        }
        public abstract void BackPropagation(Neuron neuron, List<double> outputs, double learningRate, int epoch, DelegateforWeights dw);
        
    }
    /// <summary>
    /// ////////////////////////////
    /// </summary>
    public class DefaultOptimizer : IOptimizer
    {
        public DefaultOptimizer()
        {

        }
        public DefaultOptimizer(List<Regularizations> regs) : base(regs)
        {
        }

        public override void BackPropagation(Neuron neuron, List<double> outputs, double learningRate, int epoch, DelegateforWeights dw)
        {
            double delta = neuron.Delta;
            for (int i = 0; i < neuron.WeightsCount; i++)
            {
                double output = outputs[i];
                double currentWeight = neuron.Weights[i];
                double regulator = dw(currentWeight);
                neuron.Weights[i] = currentWeight - (learningRate * delta * output + regulator);
            }
        }
    }
    /// <summary>
    /// ////////////////////////////
    /// </summary>
    public class Momentum : IOptimizer
    {
        protected double[][][] Inertions { get; set; }
        protected double Koef { get; set; }
        public Momentum(Topology topology, List<Regularizations> regs) : base(regs)
        {
            Inertions = new double[topology.Count - 1][][];
            for (int i = 0; i < topology.CollectionCounts.Count - 1; i++)
            {
                int collection = topology.CollectionCounts[i + 1];
                Inertions[i] = new double[collection][];
                int weights = topology.CollectionCounts[i];
                for (int j = 0; j < collection; j++)
                {
                    Inertions[i][j] = new double[weights];
                }
            }
            Koef = 0.9;
        }

        public Momentum(Topology topology) : base()
        {
            Inertions = new double[topology.Count - 1][][];
            for (int i = 0; i < topology.CollectionCounts.Count - 1; i++)
            {
                int collection = topology.CollectionCounts[i + 1];
                Inertions[i] = new double[collection][];
                int weights = topology.CollectionCounts[i];
                for (int j = 0; j < collection; j++)
                {
                    Inertions[i][j] = new double[weights];
                }
            }
            Koef = 0.9;
        }

        public override void BackPropagation(Neuron neuron, List<double> outputs, double learningRate, int epoch, DelegateforWeights dw)
        {
            double delta = neuron.Delta;
            int numberLayer = neuron.NumberLayer;
            for (int i = 0; i < neuron.WeightsCount; i++)
            {
                double output = outputs[i];
                double currentWeight = neuron.Weights[i];
                double regulator = dw(currentWeight);
                double inertia = Inertions[numberLayer - 1][neuron.NumberOfLayer][i] * Koef + delta * learningRate * output; // исправить error на delta в случае нестабильности!!!!!!!!!!!
                Inertions[numberLayer - 1][neuron.NumberOfLayer][i] = inertia;
                neuron.Weights[i] = currentWeight - (inertia + regulator);
            }
        }

    }
    //NOT DEFINED
    public class NesterovMomentum : Momentum
    {
        public NesterovMomentum(Topology topology, List<Regularizations> regs) : base(topology, regs)
        {

        }
        public NesterovMomentum(Topology topology) : base(topology)
        {

        }

        public override void BackPropagation(Neuron neuron, List<double> outputs, double learningRate, int epoch, DelegateforWeights dw)
        {
            double delta = neuron.Delta;
            int numberLayer = neuron.NumberLayer;
            for (int i = 0; i < neuron.WeightsCount; i++)
            {
                double output = outputs[i];
                double currentWeight = neuron.Weights[i];
                double regulator = dw(currentWeight);
                double inertia = Inertions[numberLayer - 1][neuron.NumberOfLayer][i];
                double predictedWeight = currentWeight - inertia * Koef;
                //inertia = inertia * Koef + delta * learningRate * previousOutputs[i];
                inertia = inertia * Koef + delta * learningRate * (output - Koef * inertia);
                Inertions[numberLayer - 1][neuron.NumberOfLayer][i] = inertia;
                neuron.Weights[i] = predictedWeight - (inertia + regulator);
            }
        }
    }

    public class RMSProp : IOptimizer
    {
        protected double[][][] G { get; set; }
        double E = Math.Pow(10, -8);
        double Alpha = 0.9;
        public RMSProp(Topology topology, List<Regularizations> regs) : base(regs)
        {
            G = new double[topology.Count - 1][][];
            for (int i = 0; i < topology.CollectionCounts.Count - 1; i++)
            {
                int collection = topology.CollectionCounts[i + 1];
                G[i] = new double[collection][];
                int weights = topology.CollectionCounts[i];
                for (int j = 0; j < collection; j++)
                {
                    G[i][j] = new double[weights];
                }
            }
        }
        public RMSProp(Topology topology) : base()
        {
            G = new double[topology.Count - 1][][];
            for (int i = 0; i < topology.CollectionCounts.Count - 1; i++)
            {
                int collection = topology.CollectionCounts[i + 1];
                G[i] = new double[collection][];
                int weights = topology.CollectionCounts[i];
                for (int j = 0; j < collection; j++)
                {
                    G[i][j] = new double[weights];
                }
            }
        }

        public override void BackPropagation(Neuron neuron, List<double> outputs, double learningRate, int epoch, DelegateforWeights dw)
        {
            double delta = neuron.Delta;
            int numberLayer = neuron.NumberLayer;
            for (int i = 0; i < neuron.WeightsCount; i++)
            {
                double output = outputs[i];
                double currentWeight = neuron.Weights[i];
                double regulator = dw(currentWeight);
                double g = G[numberLayer - 1][neuron.NumberOfLayer][i];
                g = Alpha * g + (1 - Alpha) * Math.Pow(delta * output, 2);
                G[numberLayer - 1][neuron.NumberOfLayer][i] = g;
                neuron.Weights[i] = currentWeight - (((delta * output * learningRate) / (Math.Sqrt(g) + E)) + regulator);
            }
        }
    }

    public class Adagrad : IOptimizer
    {
        protected double[][][] G { get; set; }
        double E = Math.Pow(10, -9);
        public Adagrad(Topology topology, List<Regularizations> regs) : base(regs)
        {
            G = new double[topology.Count - 1][][];
            for (int i = 0; i < topology.CollectionCounts.Count - 1; i++)
            {
                int collection = topology.CollectionCounts[i + 1];
                G[i] = new double[collection][];
                int weights = topology.CollectionCounts[i];
                for (int j = 0; j < collection; j++)
                {
                    G[i][j] = new double[weights];
                }
            }
        }

        public Adagrad(Topology topology) : base()
        {
            G = new double[topology.Count - 1][][];
            for (int i = 0; i < topology.CollectionCounts.Count - 1; i++)
            {
                int collection = topology.CollectionCounts[i + 1];
                G[i] = new double[collection][];
                int weights = topology.CollectionCounts[i];
                for (int j = 0; j < collection; j++)
                {
                    G[i][j] = new double[weights];
                }
            }
        }

        public override void BackPropagation(Neuron neuron, List<double> outputs, double learningRate, int epoch, DelegateforWeights dw)
        {
            double delta = neuron.Delta;
            int numberLayer = neuron.NumberLayer;
            for (int i = 0; i < neuron.WeightsCount; i++)
            {
                double output = outputs[i];
                double currentWeight = neuron.Weights[i];
                double regulator = dw(currentWeight);
                double g = G[numberLayer - 1][neuron.NumberOfLayer][i];
                double weightDelta = delta * output;
                g = g + Math.Pow(weightDelta, 2);
                neuron.Weights[i] = currentWeight - ((weightDelta * learningRate / (Math.Sqrt(g) + E)) + regulator);
                G[numberLayer - 1][neuron.NumberOfLayer][i] = g;
            }
        }
    }
    public class Adam : IOptimizer
    {
        double[][][] M { get; set; }
        double[][][] V { get; set; }
        double E = Math.Pow(10, -8);
        double Beta1 = 0.9;
        double Beta2 = 0.999;
        public Adam(Topology topology, List<Regularizations> regs) : base(regs)
        {
            M = new double[topology.Count - 1][][];
            V = new double[topology.Count - 1][][];
            for (int i = 0; i < topology.CollectionCounts.Count - 1; i++)
            {
                int collection = topology.CollectionCounts[i + 1];
                M[i] = new double[collection][];
                V[i] = new double[collection][];
                int weights = topology.CollectionCounts[i];
                for (int j = 0; j < collection; j++)
                {
                    M[i][j] = new double[weights];
                    V[i][j] = new double[weights];
                }
            }
        }

        public Adam(Topology topology) : base()
        {
            M = new double[topology.Count - 1][][];
            V = new double[topology.Count - 1][][];
            for (int i = 0; i < topology.CollectionCounts.Count - 1; i++)
            {
                int collection = topology.CollectionCounts[i + 1];
                M[i] = new double[collection][];
                V[i] = new double[collection][];
                int weights = topology.CollectionCounts[i];
                for (int j = 0; j < collection; j++)
                {
                    M[i][j] = new double[weights];
                    V[i][j] = new double[weights];
                }
            }
        }

        public override void BackPropagation(Neuron neuron, List<double> outputs, double learningRate, int epoch, DelegateforWeights dw)
        {
            double delta = neuron.Delta;
            int numberLayer = neuron.NumberLayer;
            for (int i = 0; i < neuron.WeightsCount; i++)
            {
                double output = outputs[i];
                double currentWeight = neuron.Weights[i];
                double regulator = dw(currentWeight);
                double v = V[numberLayer - 1][neuron.NumberOfLayer][i];
                double m = M[numberLayer - 1][neuron.NumberOfLayer][i];
                m = (Beta1 * m) + (1 - Beta1) * (delta * output);
                v = (Beta2 * v) + (1 - Beta2) * Math.Pow(delta * output, 2);
                double m_corr = m / (1 - Math.Pow(Beta1, epoch));
                double v_corr = v / (1 - Math.Pow(Beta2, epoch));
                neuron.Weights[i] = currentWeight - ((learningRate * m_corr) / (Math.Sqrt(v_corr) + E) + regulator);
                V[numberLayer - 1][neuron.NumberOfLayer][i] = v;
                M[numberLayer - 1][neuron.NumberOfLayer][i] = m;
            }
        }
    }
}
