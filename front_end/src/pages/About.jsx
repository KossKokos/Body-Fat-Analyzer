const About = () => {
  return (
    <div className="max-w-5xl mx-auto space-y-8">
      <section className="card">
        <div className="mb-4">
          <span className="inline-flex rounded-full bg-primary-100 px-3 py-1 text-sm font-medium text-primary-700">
            About this project
          </span>
        </div>

        <h1 className="text-4xl font-bold text-gray-900 mb-4">
          Body Fat Percentage Predictor
        </h1>

        <p className="text-lg text-gray-600 leading-8">
          This application is a portfolio project created to demonstrate
          full-stack development, machine learning integration, API design,
          frontend/backend communication, validation, and user experience
          design.
        </p>
      </section>

      <section className="card">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">
          Project purpose
        </h2>
        <div className="space-y-4 text-gray-700 leading-7">
          <p>
            The goal of this project is to showcase practical software
            engineering skills through a realistic end-to-end application.
          </p>
          <p>
            The app accepts user-entered health, fitness, and nutrition data,
            sends it to a backend prediction service, and returns an estimated
            body fat percentage. It also includes an optional feedback flow to
            demonstrate data handling, validation, API integration, and user
            interaction design.
          </p>
        </div>
      </section>

      <section className="card">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">
          What this project demonstrates
        </h2>
        <ul className="space-y-3 text-gray-700 leading-7 list-disc pl-6">
          <li>Frontend development with React and Vite</li>
          <li>Backend API integration with FastAPI</li>
          <li>Structured form validation and error handling</li>
          <li>Prediction request/response flow</li>
          <li>Feedback collection and submission logic</li>
          <li>Reusable UI components and state management</li>
          <li>Production-minded structure and clean user experience</li>
        </ul>
      </section>

      <section className="card">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">
          Tech stack
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-gray-700">
          <div className="rounded-xl border border-gray-200 p-4 bg-gray-50">
            <h3 className="font-semibold text-gray-900 mb-2">Frontend</h3>
            <p>React, JavaScript, Vite, React Hook Form, Yup, Axios</p>
          </div>

          <div className="rounded-xl border border-gray-200 p-4 bg-gray-50">
            <h3 className="font-semibold text-gray-900 mb-2">Backend</h3>
            <p>FastAPI, PostgreSQL, Alembic, TensorFlow, Docker</p>
          </div>
        </div>
      </section>

      <section className="card border border-amber-200 bg-amber-50">
        <h2 className="text-2xl font-bold text-amber-900 mb-4">
          Important disclaimer
        </h2>
        <div className="space-y-4 text-amber-900 leading-7">
          <p>
            This app is a software project for portfolio and demonstration
            purposes only.
          </p>
          <p>
            It is not a medical product, does not provide medical advice, and
            should not be used for diagnosis, treatment, or healthcare
            decision-making.
          </p>
          <p>
            Any prediction shown by the app is an estimate intended to
            demonstrate technical implementation, not a clinical result.
          </p>
        </div>
      </section>

      <section className="card">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">
          Feedback in this project
        </h2>
        <div className="space-y-4 text-gray-700 leading-7">
          <p>
            The optional feedback flow is included to demonstrate a more
            complete product experience. It allows users to rate the prediction,
            provide comments, and optionally share additional information.
          </p>
          <p>
            This part of the project exists to show how a frontend can collect
            structured feedback and send it to a backend service cleanly and
            safely.
          </p>
        </div>
      </section>

      <section className="card">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">
          Portfolio context
        </h2>
        <div className="space-y-4 text-gray-700 leading-7">
          <p>
            This hosted version is intended to help employers, recruiters, and
            reviewers see the project working in practice.
          </p>
          <p>
            It should be understood as a demonstration of development skills,
            not as a public consumer-facing health platform.
          </p>
        </div>
      </section>
    </div>
  );
};

export default About;