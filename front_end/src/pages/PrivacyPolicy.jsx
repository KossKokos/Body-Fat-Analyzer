const PrivacyPolicy = () => {
  return (
    <div className="max-w-5xl mx-auto space-y-8">
      <section className="card">
        <span className="inline-flex rounded-full bg-primary-100 px-3 py-1 text-sm font-medium text-primary-700 mb-4">
          Privacy Policy
        </span>

        <h1 className="text-4xl font-bold text-gray-900 mb-4">
          Privacy Policy
        </h1>

        <p className="text-gray-600 leading-7">
          Last updated: April 22, 2026
        </p>

        <p className="text-lg text-gray-700 leading-8 mt-4">
          This page explains how information may be handled when using this Body
          Fat Percentage Predictor project.
        </p>

        <p className="text-gray-700 leading-7 mt-4">
          This app is a personal software project and portfolio demonstration. It
          is provided to showcase full-stack engineering, machine learning
          integration, and frontend/backend development. It is not intended to
          operate as a public medical or healthcare service.
        </p>
      </section>

      <section className="card border border-amber-200 bg-amber-50">
        <h2 className="text-2xl font-bold text-amber-900 mb-4">
          Important note
        </h2>
        <p className="text-amber-900 leading-7">
          Please do not submit sensitive real-world personal or medical
          information unless you are comfortable doing so. This project is for
          demonstration purposes only.
        </p>
      </section>

      <section className="card">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">
          1. What information may be processed
        </h2>
        <div className="space-y-4 text-gray-700 leading-7">
          <p>
            If you use the prediction form, the app may process the values you
            enter, such as age, weight, height, heart-rate information, workout
            information, nutrition-related values, and similar form inputs needed
            to generate a prediction.
          </p>
          <p>
            The app may also process generated outputs such as estimated body fat
            percentage, result classification, timestamps, and related technical
            metadata needed for app operation.
          </p>
          <p>
            If you submit optional feedback, the app may process the rating,
            comment, whether the prediction felt close, any actual body fat
            percentage you choose to provide, and whether you consented to your
            feedback being used for future improvement work inside the project.
          </p>
        </div>
      </section>

      <section className="card">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">
          2. Why this information may be used
        </h2>
        <ul className="space-y-3 text-gray-700 leading-7 list-disc pl-6">
          <li>To generate prediction results inside the demo application.</li>
          <li>To support debugging, testing, and maintenance of the project.</li>
          <li>To review optional user feedback about the app experience.</li>
          <li>
            To improve the project in future development work where optional
            consent has been given.
          </li>
        </ul>
      </section>

      <section className="card border border-blue-200 bg-blue-50">
        <h2 className="text-2xl font-bold text-blue-900 mb-4">
          3. Portfolio/demo status
        </h2>
        <div className="space-y-4 text-blue-900 leading-7">
          <p>
            This app is hosted to demonstrate technical skills and project
            quality to potential employers, clients, or reviewers.
          </p>
          <p>
            It is not presented as a consumer-facing production health product,
            medical device, or professional assessment tool.
          </p>
        </div>
      </section>

      <section className="card">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">
          4. Data retention
        </h2>
        <p className="text-gray-700 leading-7">
          Information entered into the app may be stored for project functionality,
          testing, review, debugging, and demonstration purposes. Data should not
          be kept longer than reasonably necessary for those purposes.
        </p>
      </section>

      <section className="card">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">
          5. Data sharing
        </h2>
        <p className="text-gray-700 leading-7">
          This project is not intended to sell personal data. Data may be handled
          through the technical infrastructure used to host or run the project,
          such as hosting, database, monitoring, or deployment services, where
          applicable.
        </p>
      </section>

      <section className="card">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">
          6. Security
        </h2>
        <p className="text-gray-700 leading-7">
          Reasonable care may be taken to secure the application and its data, but
          no online system can guarantee absolute security. Because this is a
          project application, users should avoid treating it like a production
          medical platform.
        </p>
      </section>

      <section className="card border border-amber-200 bg-amber-50">
        <h2 className="text-2xl font-bold text-amber-900 mb-4">
          7. Not medical advice
        </h2>
        <p className="text-amber-900 leading-7">
          This project does not provide medical advice, diagnosis, treatment, or
          clinical measurement. Any prediction shown by the app is an estimate for
          demonstration purposes only.
        </p>
      </section>

      <section className="card">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">
          8. Contact
        </h2>
        <p className="text-gray-700 leading-7">
          If you have questions about this project or this page, contact:
        </p>

        <div className="mt-4 rounded-xl border border-dashed border-gray-300 p-4 bg-gray-50">
          <p className="text-sm text-gray-700">
            Email: <span className="font-medium">Kostiantyn.Pereimybida@outlook.com</span>
          </p>
        </div>
      </section>
    </div>
  );
};

export default PrivacyPolicy;