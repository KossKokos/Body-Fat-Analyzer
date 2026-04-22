const TermsOfUse = () => {
  return (
    <div className="max-w-5xl mx-auto space-y-8">
      <section className="card">
        <span className="inline-flex rounded-full bg-primary-100 px-3 py-1 text-sm font-medium text-primary-700 mb-4">
          Terms of Use
        </span>

        <h1 className="text-4xl font-bold text-gray-900 mb-4">
          Terms of Use
        </h1>

        <p className="text-gray-600 leading-7">
          Last updated: April 22, 2026
        </p>

        <p className="text-lg text-gray-700 leading-8 mt-4">
          These Terms of Use apply to this Body Fat Percentage Predictor project.
        </p>

        <p className="text-gray-700 leading-7 mt-4">
          This app is a portfolio and demonstration project created to showcase
          software engineering, machine learning integration, API design, and
          frontend/backend development skills.
        </p>
      </section>

      <section className="card border border-blue-200 bg-blue-50">
        <h2 className="text-2xl font-bold text-blue-900 mb-4">
          1. Project/demo status
        </h2>
        <p className="text-blue-900 leading-7">
          This app is not intended to operate as a public consumer service,
          medical platform, clinical assessment tool, or professional body
          composition service. It is provided primarily for project review and
          demonstration purposes.
        </p>
      </section>

      <section className="card border border-amber-200 bg-amber-50">
        <h2 className="text-2xl font-bold text-amber-900 mb-4">
          2. No medical advice
        </h2>
        <div className="space-y-4 text-amber-900 leading-7">
          <p>
            The app is for informational and demonstration purposes only.
          </p>
          <p>
            It does not provide medical advice, diagnosis, treatment, or
            professional health assessment.
          </p>
          <p>
            You should not rely on the app for healthcare decisions, diagnosis,
            emergency use, or treatment planning.
          </p>
        </div>
      </section>

      <section className="card">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">
          3. Use of the app
        </h2>
        <ul className="space-y-3 text-gray-700 leading-7 list-disc pl-6">
          <li>
            You may view and use the app for demonstration, evaluation, and
            portfolio-review purposes.
          </li>
          <li>
            You should not use the app as a substitute for a real medical,
            fitness, or clinical service.
          </li>
          <li>
            You should not submit unlawful, harmful, abusive, or deliberately
            misleading input.
          </li>
          <li>
            You should not attempt to interfere with the app, bypass protections,
            or gain unauthorized access.
          </li>
        </ul>
      </section>

      <section className="card">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">
          4. Accuracy and limitations
        </h2>
        <p className="text-gray-700 leading-7">
          Predictions produced by the app are estimates only. No guarantee is made
          that any result is accurate, complete, clinically valid, reliable, or
          suitable for your circumstances.
        </p>
      </section>

      <section className="card">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">
          5. Feedback
        </h2>
        <p className="text-gray-700 leading-7">
          If you choose to submit feedback, that feedback may be reviewed for
          project evaluation and future improvement work. If the app asks for
          optional consent to use feedback for improving future predictions, that
          choice should control such use.
        </p>
      </section>

      <section className="card">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">
          6. Availability
        </h2>
        <p className="text-gray-700 leading-7">
          This project may be changed, interrupted, limited, or removed at any
          time without notice. Continuous availability is not guaranteed.
        </p>
      </section>

      <section className="card">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">
          7. Intellectual property
        </h2>
        <p className="text-gray-700 leading-7">
          Unless otherwise stated, the app, its design, code, structure, and
          branding belong to the project owner or are used with permission.
        </p>
      </section>

      <section className="card">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">
          8. Disclaimer and liability
        </h2>
        <div className="space-y-4 text-gray-700 leading-7">
          <p>
            This project is provided on an “as is” and “as available” basis.
          </p>
          <p>
            To the fullest extent permitted by law, no warranty is given that the
            app will be uninterrupted, error-free, secure, accurate, or suitable
            for any particular purpose.
          </p>
          <p>
            The project owner is not responsible for decisions made based on the
            app’s outputs or for loss resulting from reliance on this project,
            except where liability cannot legally be excluded.
          </p>
        </div>
      </section>

      <section className="card">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">
          9. Changes to these terms
        </h2>
        <p className="text-gray-700 leading-7">
          These Terms of Use may be updated from time to time. Continued use of
          the app after changes are posted means you accept the revised version.
        </p>
      </section>

      <section className="card">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">
          10. Contact
        </h2>
        <p className="text-gray-700 leading-7">
          If you have questions about this project or these terms, contact:
        </p>

        <div className="mt-4 rounded-xl border border-dashed border-gray-300 p-4 bg-gray-50">
          <p className="text-sm text-gray-700">
            Email: <span className="font-medium">support@example.com</span>
          </p>
        </div>
      </section>
    </div>
  );
};

export default TermsOfUse;