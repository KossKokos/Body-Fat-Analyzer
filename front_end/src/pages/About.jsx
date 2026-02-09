const About = () => {
  return (
    <div className="max-w-4xl mx-auto">
      <div className="card">
        <h1 className="text-3xl font-bold text-gray-900 mb-6">About This Tool</h1>
        
        <div className="space-y-6">
          <section>
            <h2 className="text-2xl font-semibold text-gray-800 mb-3">How It Works</h2>
            <p className="text-gray-600">
              This tool uses a machine learning model trained on thousands of body composition 
              measurements to predict your body fat percentage. The model analyzes multiple 
              factors including weight, height, age, activity level, and calorie intake to 
              provide an accurate estimate.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold text-gray-800 mb-3">The Technology</h2>
            <p className="text-gray-600">
              Our prediction system uses a two-stage approach:
            </p>
            <ul className="list-disc pl-5 mt-2 text-gray-600 space-y-2">
              <li><strong>Classification:</strong> First, determines your fat category (low, moderate, high)</li>
              <li><strong>Regression:</strong> Then, calculates the exact fat percentage within that category</li>
            </ul>
          </section>

          <section>
            <h2 className="text-2xl font-semibold text-gray-800 mb-3">Accuracy Note</h2>
            <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
              <p className="text-yellow-800">
                <strong>Important:</strong> This tool provides estimates based on statistical models. 
                For precise body composition analysis, consult with healthcare professionals 
                who can use methods like DEXA scans or hydrostatic weighing.
              </p>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
};

export default About;