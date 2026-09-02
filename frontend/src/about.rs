use leptos::prelude::*;

/// The About section under the board. Every engine renders the same shape, so
/// the prose lives in `Model` and only the two child views differ.
#[component]
pub fn AboutModel(
    title: &'static str,
    intro: &'static str,
    how_it_works: AnyView,
    details: &'static [(&'static str, &'static str)],
    #[prop(optional_no_strip)] training: Option<AnyView>,
) -> impl IntoView {
    view! {
        <section class="about-model" id="about-model" aria-labelledby="about-model-title">
            <div class="about-heading">
                <div>
                    <p class="eyebrow">"ABOUT THE ENGINE"</p>
                    <h2 id="about-model-title">{title}</h2>
                </div>
                <p class="about-intro">{intro}</p>
            </div>

            {how_it_works}

            <div class="about-details">
                {details
                    .iter()
                    .map(|(heading, body)| {
                        view! {
                            <article>
                                <h3>{*heading}</h3>
                                <p>{*body}</p>
                            </article>
                        }
                    })
                    .collect_view()}
            </div>

            {training}
        </section>
    }
}
