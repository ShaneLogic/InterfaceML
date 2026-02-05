/* InterfaceML docs search and navigation */

document.addEventListener('DOMContentLoaded', () => {
    const input = document.getElementById('doc-search');
    if (!input) {
        return;
    }

    const clearBtn = document.getElementById('search-clear');
    const countEl = document.getElementById('search-count');
    const resultsEl = document.getElementById('search-results');
    const chips = document.querySelectorAll('[data-search-chip]');

    const normalize = (text) => (text || '')
        .toLowerCase()
        .replace(/\s+/g, ' ')
        .trim();

    const tokenize = (query) => normalize(query)
        .split(' ')
        .filter(Boolean);

    const sections = Array.from(document.querySelectorAll('[data-doc-section]')).map((section) => {
        const items = Array.from(section.querySelectorAll('[data-doc-item]')).map((item) => ({
            el: item,
            text: normalize(item.dataset.docSearch || item.textContent)
        }));
        const title = section.dataset.docTitle || (section.querySelector('h2, h3')?.textContent || 'Section');
        const id = section.id || title.toLowerCase().replace(/\s+/g, '-');
        const pinned = section.dataset.docPinned === 'true';
        return {
            section,
            items,
            title,
            id,
            pinned
        };
    });

    const setResults = (matches, query) => {
        if (!query) {
            countEl.textContent = 'All sections';
            resultsEl.innerHTML = '';
            return;
        }

        if (matches.length === 0) {
            countEl.textContent = '0 sections';
            resultsEl.innerHTML = '<span>No matches yet. Try a shorter query.</span>';
            return;
        }

        countEl.textContent = `${matches.length} section${matches.length === 1 ? '' : 's'}`;
        resultsEl.innerHTML = matches.slice(0, 12).map((match) => (
            `<a href="#${match.id}">${match.title} (${match.count})</a>`
        )).join('');
    };

    const update = () => {
        const query = normalize(input.value);
        const tokens = tokenize(query);
        const matches = [];

        sections.forEach((section) => {
            if (!query) {
                section.section.classList.remove('is-hidden');
                section.items.forEach((item) => item.el.classList.remove('is-hidden'));
                return;
            }

            if (section.pinned) {
                section.section.classList.remove('is-hidden');
                section.items.forEach((item) => item.el.classList.remove('is-hidden'));
                return;
            }

            let matchCount = 0;
            section.items.forEach((item) => {
                const isMatch = tokens.every((token) => item.text.includes(token));
                item.el.classList.toggle('is-hidden', !isMatch);
                if (isMatch) {
                    matchCount += 1;
                }
            });

            const showSection = matchCount > 0;
            section.section.classList.toggle('is-hidden', !showSection);

            if (matchCount > 0) {
                matches.push({
                    title: section.title,
                    id: section.id,
                    count: matchCount
                });
            }
        });

        setResults(matches, query);
    };

    input.addEventListener('input', update);
    clearBtn?.addEventListener('click', () => {
        input.value = '';
        update();
        input.focus();
    });

    chips.forEach((chip) => {
        chip.addEventListener('click', () => {
            input.value = chip.dataset.searchChip || '';
            update();
            input.focus();
        });
    });

    update();
});
